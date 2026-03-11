#include "llama-self-state.h"

#include "llama-context.h"
#include "llama-hard-memory.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <ctime>
#include <string>
#include <utility>
#include <unordered_set>

namespace {

static constexpr uint32_t LLAMA_SELF_UPDATER_VERSION = 2;
static constexpr float    LLAMA_SELF_DEFAULT_TOOL_SALIENCE_HALF_LIFE_MS = 300000.0f;
static constexpr float    LLAMA_SELF_TWO_PI = 6.28318530717958647692f;
static constexpr size_t   LLAMA_SELF_MAX_WORKING_MEMORY_ITEMS = 32;
static constexpr size_t   LLAMA_SELF_MAX_GOALS = 16;
static constexpr size_t   LLAMA_SELF_MAX_COMMITMENTS = 32;
static constexpr size_t   LLAMA_SELF_MAX_MEMORY_HANDLES = 24;
static constexpr size_t   LLAMA_SELF_MAX_TRACE_ITEMS = 256;
static constexpr uint32_t LLAMA_SELF_TRACE_MAGIC = 0x4c535354u; // LSST
static constexpr uint32_t LLAMA_SELF_TRACE_VERSION = 2;

static int64_t current_wall_clock_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

static int64_t current_monotonic_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static bool localtime_safe(std::time_t t, std::tm * out_tm) {
#if defined(_WIN32)
    return localtime_s(out_tm, &t) == 0;
#else
    return localtime_r(&t, out_tm) != nullptr;
#endif
}

static bool gmtime_safe(std::time_t t, std::tm * out_tm) {
#if defined(_WIN32)
    return gmtime_s(out_tm, &t) == 0;
#else
    return gmtime_r(&t, out_tm) != nullptr;
#endif
}

static int32_t compute_timezone_offset_minutes(int64_t wall_clock_ms) {
    const std::time_t t = (std::time_t) (wall_clock_ms / 1000);

    std::tm local_tm = {};
    std::tm gm_tm = {};
    if (!localtime_safe(t, &local_tm) || !gmtime_safe(t, &gm_tm)) {
        return 0;
    }

    local_tm.tm_isdst = -1;
    gm_tm.tm_isdst = 0;

    const std::time_t local_seconds = std::mktime(&local_tm);
    const std::time_t gm_seconds = std::mktime(&gm_tm);

    return (int32_t) std::difftime(local_seconds, gm_seconds) / 60;
}

static float clamp_unit(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

static float clamp_range(float value, float lo, float hi) {
    return std::min(hi, std::max(lo, value));
}

static std::vector<float> capture_self_register_scalars(const llama_context & ctx) {
    const int32_t count = ctx.self_state_register_count();
    std::vector<float> values(std::max(0, count), 0.0f);
    for (int32_t i = 0; i < count; ++i) {
        llama_self_register_info info = {};
        if (ctx.self_state_get_register(i, &info)) {
            values[i] = info.scalar_value;
        }
    }
    return values;
}

static llama_self_state_delta_summary summarize_self_state_delta(
        const std::vector<float> & before,
        const std::vector<float> & after,
        const llama_self_state_event & event) {
    llama_self_state_delta_summary summary = {};
    summary.role = event.role;
    summary.channel = event.channel;
    summary.flags = event.flags;

    struct delta_item {
        int32_t register_id = -1;
        float before_value = 0.0f;
        float after_value = 0.0f;
        float abs_delta = 0.0f;
    };

    std::vector<delta_item> deltas;
    const int32_t count = std::min((int32_t) before.size(), (int32_t) after.size());
    deltas.reserve(std::max(0, count));

    for (int32_t i = 0; i < count; ++i) {
        const float abs_delta = std::fabs(after[i] - before[i]);
        if (abs_delta <= 1.0e-6f) {
            continue;
        }
        summary.total_delta += abs_delta;
        summary.max_delta = std::max(summary.max_delta, abs_delta);
        deltas.push_back({
            /*.register_id =*/ i,
            /*.before_value =*/ before[i],
            /*.after_value =*/ after[i],
            /*.abs_delta =*/ abs_delta,
        });
    }

    std::sort(deltas.begin(), deltas.end(), [](const delta_item & lhs, const delta_item & rhs) {
        if (lhs.abs_delta == rhs.abs_delta) {
            return lhs.register_id < rhs.register_id;
        }
        return lhs.abs_delta > rhs.abs_delta;
    });

    summary.dimension_count = std::min<int32_t>(deltas.size(), LLAMA_SELF_STATE_MAX_DELTA_DIMS);
    for (int32_t i = 0; i < summary.dimension_count; ++i) {
        summary.dimensions[i].register_id = deltas[i].register_id;
        summary.dimensions[i].before_value = deltas[i].before_value;
        summary.dimensions[i].after_value = deltas[i].after_value;
        summary.dimensions[i].abs_delta = deltas[i].abs_delta;
    }

    return summary;
}

static float sigmoid_unit(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

template<size_t N>
static float linear_probe_score(float bias, const std::array<float, N> & weights, const std::array<float, N> & features) {
    float value = bias;
    for (size_t i = 0; i < N; ++i) {
        value += weights[i] * features[i];
    }
    return sigmoid_unit(value);
}

static int64_t elapsed_or_unset(int64_t now_ms, int64_t anchor_ms) {
    return anchor_ms < 0 ? -1 : std::max<int64_t>(0, now_ms - anchor_ms);
}

static float decay_to_unit(int64_t delta_ms, float half_life_ms) {
    if (delta_ms < 0 || half_life_ms <= 0.0f) {
        return 0.0f;
    }

    const float decay = std::exp(-std::log(2.0f) * (float) delta_ms / half_life_ms);
    return clamp_unit(decay);
}

static std::string normalize_piece(const std::string & piece) {
    std::string out;
    out.reserve(piece.size());

    for (char ch : piece) {
        const unsigned char uch = (unsigned char) ch;
        if (std::isalnum(uch) || ch == '\'' || ch == '?') {
            out.push_back((char) std::tolower(uch));
        }
    }

    return out;
}

static bool contains_any(const std::string & piece, const char * const * patterns, size_t n_patterns) {
    for (size_t i = 0; i < n_patterns; ++i) {
        if (piece.find(patterns[i]) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static llama_self_register_updater_rule make_rule(
        int32_t register_id,
        uint32_t phase_mask,
        float baseline,
        float rise_gain,
        float fall_gain,
        float baseline_pull,
        std::initializer_list<std::pair<int32_t, float>> feature_terms,
        std::initializer_list<std::pair<int32_t, float>> source_terms = {}) {
    llama_self_register_updater_rule rule = {};
    rule.register_id = register_id;
    rule.phase_mask = phase_mask;
    rule.baseline = baseline;
    rule.rise_gain = rise_gain;
    rule.fall_gain = fall_gain;
    rule.baseline_pull = baseline_pull;

    for (size_t i = 0; i < LLAMA_SELF_MAX_UPDATER_RULE_TERMS; ++i) {
        rule.feature_ids[i] = LLAMA_SELF_UPDATER_FEATURE_NONE;
    }
    for (size_t i = 0; i < LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS; ++i) {
        rule.source_register_ids[i] = -1;
    }

    size_t idx = 0;
    for (const auto & term : feature_terms) {
        if (idx >= LLAMA_SELF_MAX_UPDATER_RULE_TERMS) {
            break;
        }
        rule.feature_ids[idx] = term.first;
        rule.feature_weights[idx] = term.second;
        ++idx;
    }

    idx = 0;
    for (const auto & term : source_terms) {
        if (idx >= LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS) {
            break;
        }
        rule.source_register_ids[idx] = term.first;
        rule.source_register_weights[idx] = term.second;
        ++idx;
    }

    return rule;
}

static std::array<float, 32> build_event_sketch(const llama_self_state_event & event) {
    std::array<float, 32> sketch = {};

    if (!event.tokens || event.n_tokens == 0) {
        return sketch;
    }

    for (size_t i = 0; i < event.n_tokens; ++i) {
        const uint32_t token = (uint32_t) event.tokens[i];
        const size_t dim = token % sketch.size();
        const float sign = (token & 1u) ? -1.0f : 1.0f;
        sketch[dim] += sign;
    }

    float norm = 0.0f;
    for (float value : sketch) {
        norm += value * value;
    }
    norm = std::sqrt(norm);

    if (norm > 0.0f) {
        for (float & value : sketch) {
            value /= norm;
        }
    }

    return sketch;
}

static std::array<float, 32> build_token_sketch(const llama_token * tokens, size_t n_tokens) {
    llama_self_state_event event = {
        /*.tokens =*/ tokens,
        /*.n_tokens =*/ n_tokens,
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ 0,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 0.0f,
    };
    return build_event_sketch(event);
}

static float sketch_similarity(const std::array<float, 32> & lhs, const std::array<float, 32> & rhs) {
    float dot = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        dot += lhs[i] * rhs[i];
    }
    return std::min(1.0f, std::max(-1.0f, dot));
}

static uint32_t source_mask_for_role(const llama_self_state_event & event) {
    uint32_t mask = 0;
    switch (event.role) {
        case LLAMA_SELF_STATE_EVENT_USER:   mask |= LLAMA_SELF_SOURCE_USER_EVENT; break;
        case LLAMA_SELF_STATE_EVENT_TOOL:   mask |= LLAMA_SELF_SOURCE_TOOL_EVENT; break;
        case LLAMA_SELF_STATE_EVENT_SYSTEM: mask |= LLAMA_SELF_SOURCE_EMIT_EVENT; break;
        default:                            mask = 0; break;
    }
    if (event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) {
        mask |= LLAMA_SELF_SOURCE_COUNTERFACTUAL;
    }
    return mask;
}

static llama_self_tool_state_info build_tool_state_info(const std::vector<llama_self_tool_job> & tool_jobs) {
    llama_self_tool_state_info info = {};
    info.active_status = LLAMA_SELF_TOOL_JOB_IDLE;
    info.readiness = 0.5f;
    info.last_update_monotonic_ms = -1;

    float pending_pressure = 0.0f;
    float running_pressure = 0.0f;
    float completed_signal = 0.0f;
    float failed_signal = 0.0f;

    for (const auto & job : tool_jobs) {
        info.last_update_monotonic_ms = std::max(info.last_update_monotonic_ms, job.last_update_monotonic_ms);

        switch (job.status) {
            case LLAMA_SELF_TOOL_JOB_PENDING:
                ++info.pending_jobs;
                pending_pressure = std::max(pending_pressure, job.importance);
                info.active_status = std::max(info.active_status, (int32_t) LLAMA_SELF_TOOL_JOB_PENDING);
                break;
            case LLAMA_SELF_TOOL_JOB_RUNNING:
                ++info.running_jobs;
                running_pressure = std::max(running_pressure, job.importance);
                info.active_status = std::max(info.active_status, (int32_t) LLAMA_SELF_TOOL_JOB_RUNNING);
                break;
            case LLAMA_SELF_TOOL_JOB_COMPLETED:
                ++info.completed_jobs;
                completed_signal = std::max(completed_signal, job.importance);
                info.active_status = std::max(info.active_status, (int32_t) LLAMA_SELF_TOOL_JOB_COMPLETED);
                break;
            case LLAMA_SELF_TOOL_JOB_FAILED:
                ++info.failed_jobs;
                failed_signal = std::max(failed_signal, job.importance);
                info.active_status = std::max(info.active_status, (int32_t) LLAMA_SELF_TOOL_JOB_FAILED);
                break;
            default:
                break;
        }
    }

    if (info.failed_jobs > 0) {
        info.readiness = clamp_unit(0.15f * (1.0f - failed_signal));
    } else if (info.running_jobs > 0) {
        info.readiness = clamp_unit(0.15f + 0.25f * (1.0f - running_pressure));
    } else if (info.pending_jobs > 0) {
        info.readiness = clamp_unit(0.45f - 0.20f * pending_pressure);
    } else if (info.completed_jobs > 0) {
        info.readiness = clamp_unit(0.75f + 0.20f * completed_signal);
    }

    return info;
}

template<typename T>
static void append_bytes(std::vector<uint8_t> & out, const T & value) {
    const size_t offset = out.size();
    out.resize(offset + sizeof(T));
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

template<typename T>
static bool read_bytes(const uint8_t * src, size_t size, size_t * cursor, T * out_value) {
    if (!src || !cursor || !out_value || *cursor + sizeof(T) > size) {
        return false;
    }

    std::memcpy(out_value, src + *cursor, sizeof(T));
    *cursor += sizeof(T);
    return true;
}

static void repair_trace_item_pointers(std::vector<llama_self_trace_item> & items) {
    for (auto & item : items) {
        item.event.tokens = item.tokens.empty() ? nullptr : item.tokens.data();
        item.event.n_tokens = item.tokens.size();
    }
}

static std::array<float, 32> build_frozen_bucket_sketch(int32_t bucket_id, const llama_past_lora_bucket_stats & stats) {
    std::array<float, 32> sketch = {};
    const float values[] = {
        (float) bucket_id,
        (float) stats.version,
        stats.base_scale,
        stats.effective_scale,
        stats.gain_mean,
        stats.gain_max,
        stats.populated ? 1.0f : 0.0f,
    };

    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        const size_t dim = (bucket_id * 7 + (int32_t) i * 5) % sketch.size();
        sketch[dim] += values[i];
    }

    float norm = 0.0f;
    for (float value : sketch) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float & value : sketch) {
            value /= norm;
        }
    }

    return sketch;
}

} // namespace

std::array<llama_self_register_definition, LLAMA_SELF_REGISTER_COUNT> llama_self_state::build_definitions() {
    return {{
        { LLAMA_SELF_REGISTER_UNCERTAINTY,            "r_uncertainty",            LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_CONTRADICTION,         "r_contradiction",          LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_NOVELTY,               "r_novelty",                LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_TOPIC_SHIFT,           "r_topic_shift",            LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_GOAL_RELEVANCE,        "r_goal_relevance",         LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_SELF_RELEVANCE,        "r_self_relevance",         LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_SOCIAL_RELEVANCE,      "r_social_relevance",       LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_AFFORDANCE,            "r_affordance",             LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_BROADCAST_PRESSURE,    "r_broadcast_pressure",     LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_BROADCAST_INHIBITION,  "r_broadcast_inhibition",   LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION, "r_followup_continuation",  LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY, "r_memory_write_priority",  LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_TIME_PHASE,            "r_time_phase",             LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_TOOL_SALIENCE,         "r_tool_salience",          LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_CHANNEL_STATE,         "r_channel_state",          LLAMA_SELF_REGISTER_FAMILY_CATEGORICAL,    0.0f, 2.0f, 0.0f, LLAMA_SELF_STATE_CHANNEL_WAITING },
    }};
}

const llama_self_register_definition * llama_self_state::get_definition(int32_t register_id) {
    static const auto definitions = build_definitions();
    if (register_id < 0 || register_id >= LLAMA_SELF_REGISTER_COUNT) {
        return nullptr;
    }

    return &definitions[(size_t) register_id];
}

const char * llama_self_state::register_name(int32_t register_id) {
    const auto * def = get_definition(register_id);
    return def ? def->name : nullptr;
}

llama_self_state::llama_self_state() : definitions(build_definitions()) {
    params = llama_self_state_default_params();
    updater_program = llama_self_state_default_updater_program();

    for (size_t i = 0; i < definitions.size(); ++i) {
        const auto & def = definitions[i];
        auto & value = registers[i];
        value.scalar_value = def.default_scalar_value;
        value.categorical_value = def.default_categorical_value;
        value.confidence = 1.0f;
        value.updater_version = LLAMA_SELF_UPDATER_VERSION;
    }

    update_categorical_register(LLAMA_SELF_REGISTER_CHANNEL_STATE, channel_state, LLAMA_SELF_SOURCE_INIT);
    (void) refresh_time();
}

bool llama_self_state::configure(const llama_self_state_params & next_params) {
    params = next_params;
    if (params.tool_salience_half_life_ms <= 0.0f) {
        params.tool_salience_half_life_ms = LLAMA_SELF_DEFAULT_TOOL_SALIENCE_HALF_LIFE_MS;
    }

    params.prewrite_gain = clamp_unit(params.prewrite_gain);
    params.postwrite_gain = clamp_unit(params.postwrite_gain);
    return true;
}

bool llama_self_state::is_valid_time_point(const llama_self_state_time_point & time_point) const {
    if (time_point.wall_clock_ms < 0 || time_point.monotonic_ms < 0) {
        return false;
    }

    if (time_point.timezone_offset_minutes < -24 * 60 || time_point.timezone_offset_minutes > 24 * 60) {
        return false;
    }

    if (has_explicit_time && datetime.monotonic_ms > 0 && time_point.monotonic_ms < datetime.monotonic_ms) {
        return false;
    }

    return true;
}

bool llama_self_state::ensure_time_initialized() {
    return datetime.wall_clock_ms > 0 ? true : refresh_time();
}

bool llama_self_state::refresh_time() {
    const int64_t wall_clock_ms = current_wall_clock_ms();
    llama_self_state_time_point time_point = {
        /*.wall_clock_ms =*/ wall_clock_ms,
        /*.monotonic_ms =*/ current_monotonic_ms(),
        /*.timezone_offset_minutes =*/ compute_timezone_offset_minutes(wall_clock_ms),
    };

    return apply_time_point(time_point, LLAMA_SELF_SOURCE_TIME);
}

bool llama_self_state::set_time(llama_self_state_time_point time_point) {
    return apply_time_point(time_point, LLAMA_SELF_SOURCE_TIME | LLAMA_SELF_SOURCE_EXTERNAL_TIME);
}

bool llama_self_state::apply_time_point(const llama_self_state_time_point & time_point, uint32_t source_mask) {
    if (!is_valid_time_point(time_point)) {
        return false;
    }

    if ((source_mask & LLAMA_SELF_SOURCE_EXTERNAL_TIME) != 0 && !has_explicit_time) {
        has_explicit_time = true;
        session_start_wall_ms = time_point.wall_clock_ms;
        session_start_monotonic_ms = time_point.monotonic_ms;
        last_user_monotonic_ms = -1;
        last_tool_monotonic_ms = -1;
        last_emit_monotonic_ms = -1;
    }

    datetime.wall_clock_ms = time_point.wall_clock_ms;
    datetime.monotonic_ms = time_point.monotonic_ms;
    datetime.timezone_offset_minutes = time_point.timezone_offset_minutes;

    if (session_start_wall_ms < 0) {
        session_start_wall_ms = time_point.wall_clock_ms;
    }
    if (session_start_monotonic_ms < 0) {
        session_start_monotonic_ms = time_point.monotonic_ms;
    }

    recompute_time_surface(source_mask);
    return true;
}

void llama_self_state::recompute_time_surface(uint32_t source_mask) {
    const int64_t local_wall_ms = datetime.wall_clock_ms + (int64_t) datetime.timezone_offset_minutes * 60000;
    const std::time_t local_time = (std::time_t) (local_wall_ms / 1000);

    std::tm local_tm = {};
    if (!gmtime_safe(local_time, &local_tm)) {
        return;
    }

    datetime.local_year   = local_tm.tm_year + 1900;
    datetime.local_month  = local_tm.tm_mon + 1;
    datetime.local_day    = local_tm.tm_mday;
    datetime.local_hour   = local_tm.tm_hour;
    datetime.local_minute = local_tm.tm_min;
    datetime.local_second = local_tm.tm_sec;
    datetime.day_of_week  = local_tm.tm_wday;
    datetime.day_of_year  = local_tm.tm_yday + 1;

    const float hour_of_day = (float) datetime.local_hour +
            (float) datetime.local_minute / 60.0f +
            (float) datetime.local_second / 3600.0f;
    const float weekday_phase = (float) datetime.day_of_week / 7.0f;
    const float year_day_phase = (float) (datetime.day_of_year - 1) / 365.0f;

    datetime.hour_sin     = std::sin(LLAMA_SELF_TWO_PI * hour_of_day / 24.0f);
    datetime.hour_cos     = std::cos(LLAMA_SELF_TWO_PI * hour_of_day / 24.0f);
    datetime.weekday_sin  = std::sin(LLAMA_SELF_TWO_PI * weekday_phase);
    datetime.weekday_cos  = std::cos(LLAMA_SELF_TWO_PI * weekday_phase);
    datetime.year_day_sin = std::sin(LLAMA_SELF_TWO_PI * year_day_phase);
    datetime.year_day_cos = std::cos(LLAMA_SELF_TWO_PI * year_day_phase);

    datetime.delta_since_last_user_ms = elapsed_or_unset(datetime.monotonic_ms, last_user_monotonic_ms);
    datetime.delta_since_last_tool_event_ms = elapsed_or_unset(datetime.monotonic_ms, last_tool_monotonic_ms);
    datetime.delta_since_last_emit_ms = elapsed_or_unset(datetime.monotonic_ms, last_emit_monotonic_ms);
    datetime.session_age_ms = elapsed_or_unset(datetime.monotonic_ms, session_start_monotonic_ms);

    const float minute_of_day = (float) datetime.local_hour * 60.0f + (float) datetime.local_minute + (float) datetime.local_second / 60.0f;
    update_scalar_register(LLAMA_SELF_REGISTER_TIME_PHASE, clamp_unit(minute_of_day / (24.0f * 60.0f)), source_mask);

    const float tool_salience = decay_to_unit(datetime.delta_since_last_tool_event_ms, params.tool_salience_half_life_ms);
    update_scalar_register(LLAMA_SELF_REGISTER_TOOL_SALIENCE, tool_salience, source_mask | (tool_salience > 0.0f ? LLAMA_SELF_SOURCE_TOOL_EVENT : 0));
}

float llama_self_state::current_scalar_register(int32_t register_id) const {
    if (!get_definition(register_id)) {
        return 0.0f;
    }

    return registers[(size_t) register_id].scalar_value;
}

void llama_self_state::update_scalar_register(int32_t register_id, float value, uint32_t source_mask) {
    const auto * def = get_definition(register_id);
    if (!def) {
        return;
    }

    auto & reg = registers[(size_t) register_id];
    reg.scalar_value = clamp_range(value, def->value_min, def->value_max);
    reg.last_update_wall_ms = datetime.wall_clock_ms;
    reg.last_update_monotonic_ms = datetime.monotonic_ms;
    reg.source_mask = source_mask;
    reg.confidence = 1.0f;
    reg.updater_version = updater_program.version ? updater_program.version : LLAMA_SELF_UPDATER_VERSION;
    reg.dirty = true;
}

void llama_self_state::blend_scalar_register(int32_t register_id, float target, float gain, uint32_t source_mask) {
    const float current = current_scalar_register(register_id);
    const float blended = clamp_unit(current + gain * (clamp_unit(target) - current));
    update_scalar_register(register_id, blended, source_mask);
}

bool llama_self_state::validate_updater_program(const llama_self_updater_program & program) const {
    if (program.version == 0 || program.rule_count > LLAMA_SELF_MAX_UPDATER_RULES) {
        return false;
    }

    const float * scalar_fields[] = {
        &program.memory_novelty_weight,
        &program.memory_working_similarity_weight,
        &program.memory_handle_similarity_weight,
        &program.memory_uncertainty_weight,
        &program.memory_contradiction_weight,
        &program.memory_handle_variance_weight,
        &program.broadcast_social_weight,
        &program.broadcast_contradiction_weight,
        &program.broadcast_uncertainty_weight,
        &program.broadcast_tool_pending_weight,
        &program.broadcast_tool_unready_weight,
        &program.broadcast_failure_weight,
        &program.broadcast_question_weight,
        &program.broadcast_goal_weight,
        &program.repair_emit_threshold,
        &program.repair_dissatisfaction_floor,
        &program.repair_recent_user_valence_floor,
        &program.repair_inhibition_max,
        &program.repair_admission_floor,
        &program.repair_admission_weight,
    };
    for (const float * field : scalar_fields) {
        if (!std::isfinite(*field)) {
            return false;
        }
    }

    const uint32_t valid_phase_mask = LLAMA_SELF_UPDATER_PHASE_PREWRITE | LLAMA_SELF_UPDATER_PHASE_POSTWRITE;

    for (uint32_t i = 0; i < program.rule_count; ++i) {
        const auto & rule = program.rules[i];
        const auto * def = get_definition(rule.register_id);
        if (!def || def->family != LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR) {
            return false;
        }
        if ((rule.phase_mask & valid_phase_mask) == 0 || (rule.phase_mask & ~valid_phase_mask) != 0) {
            return false;
        }
        if (!std::isfinite(rule.baseline) || !std::isfinite(rule.rise_gain) ||
            !std::isfinite(rule.fall_gain) || !std::isfinite(rule.baseline_pull)) {
            return false;
        }
        if (rule.rise_gain < 0.0f || rule.rise_gain > 1.0f ||
            rule.fall_gain < 0.0f || rule.fall_gain > 1.0f ||
            rule.baseline_pull < 0.0f || rule.baseline_pull > 1.0f) {
            return false;
        }

        for (size_t term = 0; term < LLAMA_SELF_MAX_UPDATER_RULE_TERMS; ++term) {
            const int32_t feature_id = rule.feature_ids[term];
            if (feature_id == LLAMA_SELF_UPDATER_FEATURE_NONE) {
                continue;
            }
            if (feature_id < LLAMA_SELF_UPDATER_FEATURE_NOVELTY ||
                feature_id > LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_COMPLETED ||
                !std::isfinite(rule.feature_weights[term])) {
                return false;
            }
        }

        for (size_t term = 0; term < LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS; ++term) {
            const int32_t source_register_id = rule.source_register_ids[term];
            if (source_register_id < 0) {
                continue;
            }
            const auto * source_def = get_definition(source_register_id);
            if (!source_def || source_def->family != LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR ||
                !std::isfinite(rule.source_register_weights[term])) {
                return false;
            }
        }
    }

    return true;
}

float llama_self_state::updater_feature_value(
        int32_t feature_id,
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features) const {
    switch (feature_id) {
        case LLAMA_SELF_UPDATER_FEATURE_NOVELTY: return features.novelty;
        case LLAMA_SELF_UPDATER_FEATURE_TOPIC_SHIFT: return features.topic_shift;
        case LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY: return features.goal_top_similarity;
        case LLAMA_SELF_UPDATER_FEATURE_COMMITMENT_SIMILARITY: return features.commitment_top_similarity;
        case LLAMA_SELF_UPDATER_FEATURE_IDENTITY_SIMILARITY: return features.identity_similarity;
        case LLAMA_SELF_UPDATER_FEATURE_SELF_REFERENCE: return features.self_reference_ratio;
        case LLAMA_SELF_UPDATER_FEATURE_NEGATION_RATIO: return features.negation_ratio;
        case LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY_LEXICAL: return features.uncertainty_lexical_ratio;
        case LLAMA_SELF_UPDATER_FEATURE_ERROR_RATIO: return features.error_ratio;
        case LLAMA_SELF_UPDATER_FEATURE_RECENCY_USER: return features.recency_user;
        case LLAMA_SELF_UPDATER_FEATURE_RECENCY_TOOL: return features.recency_tool;
        case LLAMA_SELF_UPDATER_FEATURE_RECENCY_EMIT: return features.recency_emit;
        case LLAMA_SELF_UPDATER_FEATURE_SOCIAL_FAMILIARITY: return features.social_familiarity;
        case LLAMA_SELF_UPDATER_FEATURE_SOCIAL_TRUST: return features.social_trust;
        case LLAMA_SELF_UPDATER_FEATURE_SOCIAL_RECIPROCITY: return features.social_reciprocity;
        case LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND: return features.social_bond_strength;
        case LLAMA_SELF_UPDATER_FEATURE_TOOL_READINESS: return features.tool_readiness_score;
        case LLAMA_SELF_UPDATER_FEATURE_TOOL_PENDING_PRESSURE: return features.tool_pending_pressure;
        case LLAMA_SELF_UPDATER_FEATURE_DECODER_ENTROPY: return features.decoder_entropy;
        case LLAMA_SELF_UPDATER_FEATURE_DECODER_TOP_MARGIN: return features.decoder_top_margin;
        case LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION: return features.contradiction_score;
        case LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY: return features.uncertainty_score;
        case LLAMA_SELF_UPDATER_FEATURE_MEMORY_WRITE_PRESSURE: return features.memory_write_pressure;
        case LLAMA_SELF_UPDATER_FEATURE_BROADCAST_PRESSURE_HINT: return features.broadcast_pressure_hint;
        case LLAMA_SELF_UPDATER_FEATURE_BROADCAST_INHIBITION_HINT: return features.broadcast_inhibition_hint;
        case LLAMA_SELF_UPDATER_FEATURE_FOLLOWUP_HINT: return features.followup_hint;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_USER: return event.role == LLAMA_SELF_STATE_EVENT_USER ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_TOOL: return event.role == LLAMA_SELF_STATE_EVENT_TOOL ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_SYSTEM: return event.role == LLAMA_SELF_STATE_EVENT_SYSTEM ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_CHANNEL_PRIMARY: return event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_CHANNEL_COUNTERFACTUAL: return event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_ADMITTED: return (event.flags & LLAMA_SELF_STATE_EVENT_ADMITTED) ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_FAILED: return (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_COMPLETED: return (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED) ? 1.0f : 0.0f;
        default: return 0.0f;
    }
}

bool llama_self_state::apply_register_update_rules(
        uint32_t phase_mask,
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features,
        uint32_t source_mask) {
    std::array<float, LLAMA_SELF_REGISTER_COUNT> snapshot = {};
    for (size_t i = 0; i < snapshot.size(); ++i) {
        snapshot[i] = registers[i].scalar_value;
    }

    for (uint32_t i = 0; i < updater_program.rule_count; ++i) {
        const auto & rule = updater_program.rules[i];
        if ((rule.phase_mask & phase_mask) == 0) {
            continue;
        }

        const auto * def = get_definition(rule.register_id);
        if (!def || def->family != LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR) {
            continue;
        }

        float target = rule.baseline;
        for (size_t term = 0; term < LLAMA_SELF_MAX_UPDATER_RULE_TERMS; ++term) {
            const int32_t feature_id = rule.feature_ids[term];
            if (feature_id == LLAMA_SELF_UPDATER_FEATURE_NONE) {
                continue;
            }
            target += rule.feature_weights[term] * updater_feature_value(feature_id, event, features);
        }
        for (size_t term = 0; term < LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS; ++term) {
            const int32_t source_register_id = rule.source_register_ids[term];
            if (source_register_id < 0 || (size_t) source_register_id >= snapshot.size()) {
                continue;
            }
            target += rule.source_register_weights[term] * snapshot[(size_t) source_register_id];
        }

        const float bounded_target = clamp_range(target, def->value_min, def->value_max);
        const float current = snapshot[(size_t) rule.register_id];
        const float delta = bounded_target - current;
        const float gain = delta >= 0.0f ? rule.rise_gain : rule.fall_gain;
        const float next = clamp_range(
                current + gain * delta + rule.baseline_pull * (rule.baseline - current),
                def->value_min,
                def->value_max);
        update_scalar_register(rule.register_id, next, source_mask);
    }

    return true;
}

void llama_self_state::update_categorical_register(int32_t register_id, int32_t value, uint32_t source_mask) {
    if (!get_definition(register_id)) {
        return;
    }

    auto & reg = registers[(size_t) register_id];
    reg.categorical_value = value;
    reg.last_update_wall_ms = datetime.wall_clock_ms;
    reg.last_update_monotonic_ms = datetime.monotonic_ms;
    reg.source_mask = source_mask;
    reg.confidence = 1.0f;
    reg.updater_version = updater_program.version ? updater_program.version : LLAMA_SELF_UPDATER_VERSION;
    reg.dirty = true;
}

bool llama_self_state::get_datetime(llama_self_state_datetime * out_info) const {
    if (!out_info) {
        return false;
    }

    *out_info = datetime;
    return true;
}

int32_t llama_self_state::register_count() const {
    return LLAMA_SELF_REGISTER_COUNT;
}

bool llama_self_state::get_register(int32_t register_id, llama_self_register_info * out_info) const {
    if (!out_info) {
        return false;
    }

    const auto * def = get_definition(register_id);
    if (!def) {
        return false;
    }

    const auto & reg = registers[(size_t) register_id];
    *out_info = {
        /*.register_id =*/ register_id,
        /*.family =*/ def->family,
        /*.scalar_value =*/ reg.scalar_value,
        /*.categorical_value =*/ reg.categorical_value,
        /*.value_min =*/ def->value_min,
        /*.value_max =*/ def->value_max,
        /*.confidence =*/ reg.confidence,
        /*.last_update_wall_ms =*/ reg.last_update_wall_ms,
        /*.last_update_monotonic_ms =*/ reg.last_update_monotonic_ms,
        /*.source_mask =*/ reg.source_mask,
        /*.updater_version =*/ reg.updater_version,
        /*.dirty =*/ reg.dirty,
    };
    return true;
}

bool llama_self_state::set_channel_state(int32_t next_channel_state) {
    if (next_channel_state != LLAMA_SELF_STATE_CHANNEL_WAITING &&
        next_channel_state != LLAMA_SELF_STATE_CHANNEL_ACTIVE &&
        next_channel_state != LLAMA_SELF_STATE_CHANNEL_DO_NOT_INTERRUPT) {
        return false;
    }

    if (!ensure_time_initialized()) {
        return false;
    }

    channel_state = next_channel_state;
    update_categorical_register(LLAMA_SELF_REGISTER_CHANNEL_STATE, channel_state, LLAMA_SELF_SOURCE_CHANNEL);
    return true;
}

bool llama_self_state::note_user_event() {
    if (!ensure_time_initialized()) {
        return false;
    }

    last_user_monotonic_ms = datetime.monotonic_ms;
    datetime.delta_since_last_user_ms = 0;
    update_categorical_register(LLAMA_SELF_REGISTER_CHANNEL_STATE, channel_state, LLAMA_SELF_SOURCE_USER_EVENT | LLAMA_SELF_SOURCE_CHANNEL);
    return true;
}

bool llama_self_state::note_tool_event() {
    if (!ensure_time_initialized()) {
        return false;
    }

    last_tool_monotonic_ms = datetime.monotonic_ms;
    datetime.delta_since_last_tool_event_ms = 0;
    update_scalar_register(LLAMA_SELF_REGISTER_TOOL_SALIENCE, 1.0f, LLAMA_SELF_SOURCE_TOOL_EVENT);
    return true;
}

bool llama_self_state::note_emit_event() {
    if (!ensure_time_initialized()) {
        return false;
    }

    last_emit_monotonic_ms = datetime.monotonic_ms;
    datetime.delta_since_last_emit_ms = 0;
    return true;
}

bool llama_self_state::set_identity(const llama_token * tokens, size_t n_tokens) {
    if (!tokens || n_tokens == 0) {
        return false;
    }

    identity_sketch = build_token_sketch(tokens, n_tokens);
    has_identity_sketch = true;
    return true;
}

void llama_self_state::upsert_surface(
        std::vector<llama_self_sketch_surface> & surfaces,
        int32_t id,
        const std::array<float, 32> & sketch,
        float priority,
        bool unresolved) {
    auto it = std::find_if(surfaces.begin(), surfaces.end(), [id](const llama_self_sketch_surface & surface) {
        return surface.id == id;
    });

    if (it == surfaces.end()) {
        surfaces.push_back({});
        it = surfaces.end() - 1;
    }

    it->id = id;
    it->priority = clamp_unit(priority);
    it->unresolved = unresolved;
    it->last_update_monotonic_ms = datetime.monotonic_ms;
    it->sketch = sketch;
}

bool llama_self_state::upsert_goal(int32_t goal_id, const llama_token * tokens, size_t n_tokens, float priority) {
    if (goal_id < 0 || !tokens || n_tokens == 0) {
        return false;
    }

    upsert_surface(goals, goal_id, build_token_sketch(tokens, n_tokens), priority, true);
    if (goals.size() > LLAMA_SELF_MAX_GOALS) {
        std::sort(goals.begin(), goals.end(), [](const llama_self_sketch_surface & lhs, const llama_self_sketch_surface & rhs) {
            return lhs.priority > rhs.priority;
        });
        goals.resize(LLAMA_SELF_MAX_GOALS);
    }
    return true;
}

bool llama_self_state::upsert_commitment(
        int32_t commitment_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority,
        bool unresolved) {
    if (commitment_id < 0 || !tokens || n_tokens == 0) {
        return false;
    }

    upsert_surface(commitments, commitment_id, build_token_sketch(tokens, n_tokens), priority, unresolved);
    if (commitments.size() > LLAMA_SELF_MAX_COMMITMENTS) {
        std::sort(commitments.begin(), commitments.end(), [](const llama_self_sketch_surface & lhs, const llama_self_sketch_surface & rhs) {
            return lhs.priority > rhs.priority;
        });
        commitments.resize(LLAMA_SELF_MAX_COMMITMENTS);
    }
    return true;
}

int32_t llama_self_state::goal_count() const {
    return (int32_t) goals.size();
}

int32_t llama_self_state::commitment_count() const {
    return (int32_t) commitments.size();
}

int32_t llama_self_state::working_memory_count() const {
    return (int32_t) working_memory.size();
}

bool llama_self_state::upsert_memory_handle(
        int32_t handle_id,
        int32_t kind,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    if (handle_id < 0 || !tokens || n_tokens == 0 || !ensure_time_initialized()) {
        return false;
    }

    if (kind != LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER &&
        kind != LLAMA_SELF_MEMORY_HANDLE_FROZEN_BUCKET &&
        kind != LLAMA_SELF_MEMORY_HANDLE_ACTIVE_MEMORY &&
        kind != LLAMA_SELF_MEMORY_HANDLE_EXTERNAL) {
        return false;
    }

    const auto sketch = build_token_sketch(tokens, n_tokens);
    return upsert_memory_handle_sketch(handle_id, kind, sketch, priority, 1);
}

bool llama_self_state::upsert_memory_handle_sketch(
        int32_t handle_id,
        int32_t kind,
        const std::array<float, 32> & sketch,
        float priority,
        uint32_t member_count) {
    if (handle_id < 0 || !ensure_time_initialized()) {
        return false;
    }

    if (kind != LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER &&
        kind != LLAMA_SELF_MEMORY_HANDLE_FROZEN_BUCKET &&
        kind != LLAMA_SELF_MEMORY_HANDLE_ACTIVE_MEMORY &&
        kind != LLAMA_SELF_MEMORY_HANDLE_EXTERNAL) {
        return false;
    }

    auto it = std::find_if(memory_handles.begin(), memory_handles.end(), [handle_id](const llama_self_memory_handle & handle) {
        return handle.handle_id == handle_id;
    });

    if (it == memory_handles.end()) {
        memory_handles.push_back({});
        it = memory_handles.end() - 1;
    }

    it->handle_id = handle_id;
    it->kind = kind;
    it->priority = clamp_unit(priority);
    it->last_update_monotonic_ms = datetime.monotonic_ms;
    it->member_count = std::max<uint32_t>(1, member_count);
    it->centroid = sketch;

    if (memory_handles.size() > LLAMA_SELF_MAX_MEMORY_HANDLES) {
        std::sort(memory_handles.begin(), memory_handles.end(), [](const llama_self_memory_handle & lhs, const llama_self_memory_handle & rhs) {
            return lhs.priority > rhs.priority;
        });
        memory_handles.resize(LLAMA_SELF_MAX_MEMORY_HANDLES);
    }
    return true;
}

int32_t llama_self_state::memory_handle_count() const {
    return (int32_t) memory_handles.size();
}

int32_t llama_self_state::reactivation_count() const {
    return (int32_t) reactivation_priorities.size();
}

bool llama_self_state::get_reactivation(int32_t index, llama_self_reactivation_info * out_info) const {
    if (!out_info || index < 0 || (size_t) index >= reactivation_priorities.size()) {
        return false;
    }

    *out_info = reactivation_priorities[(size_t) index];
    return true;
}

bool llama_self_state::upsert_tool_job(int32_t job_id, int32_t status, float importance) {
    if (job_id < 0 || !ensure_time_initialized()) {
        return false;
    }

    if (status < LLAMA_SELF_TOOL_JOB_IDLE || status > LLAMA_SELF_TOOL_JOB_FAILED) {
        return false;
    }

    auto it = std::find_if(tool_jobs.begin(), tool_jobs.end(), [job_id](const llama_self_tool_job & job) {
        return job.job_id == job_id;
    });

    if (it == tool_jobs.end()) {
        tool_jobs.push_back({});
        it = tool_jobs.end() - 1;
        it->job_id = job_id;
    }

    it->status = status;
    it->importance = clamp_unit(importance);
    it->last_update_monotonic_ms = datetime.monotonic_ms;
    last_tool_monotonic_ms = datetime.monotonic_ms;
    datetime.delta_since_last_tool_event_ms = 0;
    refresh_tool_surface(LLAMA_SELF_SOURCE_TOOL_EVENT);
    return true;
}

bool llama_self_state::get_tool_state(llama_self_tool_state_info * out_info) const {
    if (!out_info) {
        return false;
    }

    *out_info = build_tool_state_info(tool_jobs);
    return true;
}

bool llama_self_state::get_social_state(llama_self_social_state_info * out_info) const {
    if (!out_info) {
        return false;
    }

    const float bond_strength = clamp_unit(
            0.40f * social_familiarity +
            0.35f * social_trust +
            0.25f * social_reciprocity);

    *out_info = {
        /*.familiarity =*/ social_familiarity,
        /*.trust =*/ social_trust,
        /*.reciprocity =*/ social_reciprocity,
        /*.bond_strength =*/ bond_strength,
        /*.recent_user_valence =*/ social_recent_user_valence,
        /*.dissatisfaction =*/ social_dissatisfaction,
        /*.user_turn_count =*/ social_user_turn_count,
        /*.system_turn_count =*/ social_system_turn_count,
        /*.last_update_monotonic_ms =*/ social_last_update_monotonic_ms,
    };
    return true;
}

int32_t llama_self_state::trace_count() const {
    return (int32_t) trace_items.size();
}

bool llama_self_state::clear_trace() {
    trace_items.clear();
    return true;
}

bool llama_self_state::set_updater_program(const llama_self_updater_program & program) {
    if (!validate_updater_program(program)) {
        return false;
    }

    updater_program = program;
    return true;
}

bool llama_self_state::get_updater_program(llama_self_updater_program * out_program) const {
    if (!out_program) {
        return false;
    }

    *out_program = updater_program;
    return true;
}

size_t llama_self_state::trace_export_size() const {
    size_t size = sizeof(uint32_t) * 3;

    for (const auto & item : trace_items) {
        size += sizeof(llama_self_state_time_point);
        size += sizeof(int32_t);
        size += sizeof(int32_t);
        size += sizeof(uint32_t);
        size += sizeof(float) * 2;
        size += sizeof(uint32_t);
        size += item.tokens.size() * sizeof(llama_token);
    }

    return size;
}

bool llama_self_state::trace_export(void * dst, size_t size) const {
    if (!dst || size < trace_export_size()) {
        return false;
    }

    std::vector<uint8_t> buffer;
    buffer.reserve(trace_export_size());
    append_bytes(buffer, LLAMA_SELF_TRACE_MAGIC);
    append_bytes(buffer, LLAMA_SELF_TRACE_VERSION);
    append_bytes(buffer, (uint32_t) trace_items.size());

    for (const auto & item : trace_items) {
        append_bytes(buffer, item.time_point);
        append_bytes(buffer, item.event.role);
        append_bytes(buffer, item.event.channel);
        append_bytes(buffer, item.event.flags);
        append_bytes(buffer, item.event.decoder_entropy);
        append_bytes(buffer, item.event.decoder_top_margin);
        append_bytes(buffer, (uint32_t) item.tokens.size());
        for (llama_token token : item.tokens) {
            append_bytes(buffer, token);
        }
    }

    if (buffer.size() > size) {
        return false;
    }

    std::memcpy(dst, buffer.data(), buffer.size());
    return true;
}

bool llama_self_state::trace_import(const void * src, size_t size, bool replace_existing) {
    if (!src) {
        return false;
    }

    const uint8_t * bytes = (const uint8_t *) src;
    size_t cursor = 0;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    if (!read_bytes(bytes, size, &cursor, &magic) ||
        !read_bytes(bytes, size, &cursor, &version) ||
        !read_bytes(bytes, size, &cursor, &count) ||
        magic != LLAMA_SELF_TRACE_MAGIC ||
        (version != 1 && version != LLAMA_SELF_TRACE_VERSION)) {
        return false;
    }

    std::vector<llama_self_trace_item> imported;
    imported.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        llama_self_state_time_point time_point = {};
        int32_t role = 0;
        int32_t channel = LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY;
        uint32_t flags = 0;
        float decoder_entropy = 0.0f;
        float decoder_top_margin = 0.0f;
        uint32_t n_tokens = 0;

        if (!read_bytes(bytes, size, &cursor, &time_point) ||
            !read_bytes(bytes, size, &cursor, &role) ||
            (version >= 2 && !read_bytes(bytes, size, &cursor, &channel)) ||
            !read_bytes(bytes, size, &cursor, &flags) ||
            !read_bytes(bytes, size, &cursor, &decoder_entropy) ||
            !read_bytes(bytes, size, &cursor, &decoder_top_margin) ||
            !read_bytes(bytes, size, &cursor, &n_tokens)) {
            return false;
        }

        llama_self_trace_item item = {};
        item.time_point = time_point;
        item.tokens.resize(n_tokens);
        for (uint32_t j = 0; j < n_tokens; ++j) {
            if (!read_bytes(bytes, size, &cursor, &item.tokens[(size_t) j])) {
                return false;
            }
        }

        item.event = {
            /*.tokens =*/ item.tokens.data(),
            /*.n_tokens =*/ item.tokens.size(),
            /*.role =*/ role,
            /*.channel =*/ channel,
            /*.flags =*/ flags,
            /*.decoder_entropy =*/ decoder_entropy,
            /*.decoder_top_margin =*/ decoder_top_margin,
        };
        imported.push_back(std::move(item));
    }

    if (cursor != size) {
        return false;
    }

    if (replace_existing) {
        trace_items = std::move(imported);
    } else {
        trace_items.insert(trace_items.end(), imported.begin(), imported.end());
        if (trace_items.size() > LLAMA_SELF_MAX_TRACE_ITEMS) {
            trace_items.erase(trace_items.begin(), trace_items.end() - LLAMA_SELF_MAX_TRACE_ITEMS);
        }
    }

    repair_trace_item_pointers(trace_items);

    return true;
}

void llama_self_state::reset_dynamic_state_preserve_static() {
    registers = {};
    for (size_t i = 0; i < definitions.size(); ++i) {
        const auto & def = definitions[i];
        auto & value = registers[i];
        value.scalar_value = def.default_scalar_value;
        value.categorical_value = def.default_categorical_value;
        value.confidence = 1.0f;
        value.updater_version = LLAMA_SELF_UPDATER_VERSION;
    }

    working_memory.clear();
    tool_jobs.clear();
    reactivation_priorities.clear();
    next_working_memory_event_id = 1;
    has_previous_event_sketch = false;
    previous_event_sketch = {};
    last_user_monotonic_ms = -1;
    last_tool_monotonic_ms = -1;
    last_emit_monotonic_ms = -1;
    social_last_update_monotonic_ms = -1;
    social_user_turn_count = 0;
    social_system_turn_count = 0;
    social_familiarity = 0.0f;
    social_trust = 0.5f;
    social_reciprocity = 0.5f;
    social_recent_user_valence = 0.0f;
    social_dissatisfaction = 0.0f;
    session_start_wall_ms = -1;
    session_start_monotonic_ms = -1;
    has_explicit_time = false;
    channel_state = LLAMA_SELF_STATE_CHANNEL_WAITING;
    update_categorical_register(LLAMA_SELF_REGISTER_CHANNEL_STATE, channel_state, LLAMA_SELF_SOURCE_INIT);
    recompute_time_surface(LLAMA_SELF_SOURCE_TIME);
    refresh_tool_surface(LLAMA_SELF_SOURCE_INIT | LLAMA_SELF_SOURCE_TOOL_EVENT);
}

void llama_self_state::append_trace(const llama_self_state_event & event) {
    llama_self_trace_item item = {};
    item.time_point = {
        /*.wall_clock_ms =*/ datetime.wall_clock_ms,
        /*.monotonic_ms =*/ datetime.monotonic_ms,
        /*.timezone_offset_minutes =*/ datetime.timezone_offset_minutes,
    };
    item.tokens.assign(event.tokens, event.tokens + event.n_tokens);
    item.event = event;
    item.event.tokens = item.tokens.data();
    item.event.n_tokens = item.tokens.size();
    trace_items.push_back(std::move(item));

    if (trace_items.size() > LLAMA_SELF_MAX_TRACE_ITEMS) {
        trace_items.erase(trace_items.begin());
    }
}

bool llama_self_state::replay_trace(const llama_vocab * vocab, int32_t upto_count, int32_t override_channel) {
    if (!vocab) {
        return false;
    }

    if (override_channel != -1 &&
        override_channel != LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY &&
        override_channel != LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) {
        return false;
    }

    if (upto_count < 0) {
        upto_count = (int32_t) trace_items.size();
    }

    const size_t limit = std::min(trace_items.size(), (size_t) upto_count);
    auto saved_trace = trace_items;
    repair_trace_item_pointers(saved_trace);
    trace_items.clear();
    reset_dynamic_state_preserve_static();

    for (size_t i = 0; i < limit; ++i) {
        auto item = saved_trace[i];
        if (override_channel != -1) {
            item.event.channel = override_channel;
        }
        if (!set_time(item.time_point)) {
            trace_items = saved_trace;
            return false;
        }

        llama_self_state_feature_vector prewrite = {};
        if (!build_prewrite_features(vocab, item.event, &prewrite) ||
            !apply_prewrite(item.event, prewrite)) {
            trace_items = saved_trace;
            return false;
        }

        llama_self_state_feature_vector postwrite = {};
        if (!build_postwrite_features(vocab, item.event, &postwrite) ||
            !apply_postwrite(item.event, postwrite)) {
            trace_items = saved_trace;
            return false;
        }
    }

    trace_items = saved_trace;
    return true;
}

bool llama_self_state::evaluate_counterfactual(
        const llama_vocab * vocab,
        const llama_self_updater_program & program,
        int32_t upto_count,
        int32_t replay_channel,
        llama_self_counterfactual_result * out_result) const {
    if (!out_result || !vocab || program.version == 0) {
        return false;
    }

    llama_self_state shadow = *this;
    repair_trace_item_pointers(shadow.trace_items);
    if (!shadow.set_updater_program(program) || !shadow.replay_trace(vocab, upto_count, replay_channel)) {
        return false;
    }

    *out_result = {
        /*.updater_version =*/ program.version,
        /*.replay_channel =*/ replay_channel,
        /*.replayed_events =*/ upto_count < 0 ? (int32_t) trace_items.size() : std::min((int32_t) trace_items.size(), upto_count),
        /*.working_memory_count =*/ shadow.working_memory_count(),
        /*.reactivation_count =*/ shadow.reactivation_count(),
        /*.uncertainty =*/ shadow.current_scalar_register(LLAMA_SELF_REGISTER_UNCERTAINTY),
        /*.contradiction =*/ shadow.current_scalar_register(LLAMA_SELF_REGISTER_CONTRADICTION),
        /*.memory_write_priority =*/ shadow.current_scalar_register(LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY),
        /*.broadcast_pressure =*/ shadow.current_scalar_register(LLAMA_SELF_REGISTER_BROADCAST_PRESSURE),
    };
    return true;
}

float llama_self_state::max_similarity(
        const std::vector<llama_self_sketch_surface> & surfaces,
        const std::array<float, 32> & sketch,
        bool unresolved_only) const {
    float best = 0.0f;

    for (const auto & surface : surfaces) {
        if (unresolved_only && !surface.unresolved) {
            continue;
        }

        const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, surface.sketch)));
        best = std::max(best, similarity * std::max(0.25f, surface.priority));
    }

    return clamp_unit(best);
}

void llama_self_state::working_memory_stats(
        const std::array<float, 32> & sketch,
        float * out_top_similarity,
        float * out_variance) const {
    if (!out_top_similarity || !out_variance) {
        return;
    }

    *out_top_similarity = 0.0f;
    *out_variance = 0.0f;

    if (working_memory.empty()) {
        return;
    }

    float top = 0.0f;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (const auto & item : working_memory) {
        const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, item.sketch)));
        top = std::max(top, similarity);
        sum += similarity;
        sum_sq += similarity * similarity;
    }

    const float count = (float) working_memory.size();
    const float mean = sum / count;
    *out_top_similarity = top;
    *out_variance = clamp_unit(std::max(0.0f, sum_sq / count - mean * mean));
}

void llama_self_state::memory_handle_stats(
        const std::array<float, 32> & sketch,
        float * out_top_similarity,
        float * out_variance) const {
    if (!out_top_similarity || !out_variance) {
        return;
    }

    *out_top_similarity = 0.0f;
    *out_variance = 0.0f;

    if (memory_handles.empty()) {
        return;
    }

    float top = 0.0f;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (const auto & handle : memory_handles) {
        const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, handle.centroid)));
        const float weighted = clamp_unit(similarity * std::max(0.25f, handle.priority));
        top = std::max(top, weighted);
        sum += weighted;
        sum_sq += weighted * weighted;
    }

    const float count = (float) memory_handles.size();
    const float mean = sum / count;
    *out_top_similarity = top;
    *out_variance = clamp_unit(std::max(0.0f, sum_sq / count - mean * mean));
}

void llama_self_state::admit_working_memory(
        const llama_self_state_event & event,
        const std::array<float, 32> & sketch,
        const llama_self_state_feature_vector & features) {
    llama_self_working_memory_item item = {};
    item.event_id = next_working_memory_event_id++;
    item.role = event.role;
    item.flags = event.flags;
    item.salience = clamp_unit(features.memory_write_pressure);
    item.unresolved_question = event.role == LLAMA_SELF_STATE_EVENT_USER &&
            (features.followup_hint > 0.25f || features.uncertainty_score > 0.45f);
    item.tool_affordance_hint = features.tool_pending_pressure > 0.15f ||
            features.tool_readiness_score < 0.55f ||
            current_scalar_register(LLAMA_SELF_REGISTER_AFFORDANCE) > 0.45f;
    item.admitted_monotonic_ms = datetime.monotonic_ms;
    item.sketch = sketch;
    working_memory.push_back(item);

    if (working_memory.size() > LLAMA_SELF_MAX_WORKING_MEMORY_ITEMS) {
        working_memory.erase(working_memory.begin());
    }

    bridge_working_memory_to_handles(sketch, item.salience);
}

float llama_self_state::run_contradiction_head(float analytic_score, const llama_self_state_feature_vector & features) const {
    if (params.enable_builtin_contradiction_probe) {
        static constexpr std::array<float, 12> weights = {{
            2.25f, 1.50f, 0.25f, 0.10f, 0.30f, 0.35f,
            0.90f, 0.35f, 0.40f, 0.20f, 0.10f, -0.20f,
        }};
        const std::array<float, 12> probe_features = {{
            features.negation_ratio,
            features.error_ratio,
            features.uncertainty_lexical_ratio,
            features.decoder_entropy,
            1.0f - features.decoder_top_margin,
            features.novelty,
            features.commitment_top_similarity,
            features.tool_pending_pressure,
            1.0f - features.tool_readiness_score,
            features.memory_handle_similarity_variance,
            analytic_score,
            features.social_trust,
        }};
        const float builtin_score = linear_probe_score(-1.15f, weights, probe_features);
        analytic_score = clamp_unit(0.55f * analytic_score + 0.45f * builtin_score);
    }

    if (!params.enable_learned_contradiction_head || !params.contradiction_head_callback) {
        return analytic_score;
    }

    float head_score = analytic_score;
    if (!params.contradiction_head_callback(&features, &head_score, params.contradiction_head_user_data) || !std::isfinite(head_score)) {
        return analytic_score;
    }

    return clamp_unit(head_score);
}

float llama_self_state::run_uncertainty_head(float analytic_score, const llama_self_state_feature_vector & features) const {
    if (params.enable_builtin_uncertainty_probe) {
        static constexpr std::array<float, 12> weights = {{
            0.20f, 0.55f, 2.30f, 1.85f, 1.40f, 0.30f,
            0.10f, 0.25f, 0.35f, 0.25f, 0.10f, -0.15f,
        }};
        const std::array<float, 12> probe_features = {{
            features.negation_ratio,
            features.error_ratio,
            features.uncertainty_lexical_ratio,
            features.decoder_entropy,
            1.0f - features.decoder_top_margin,
            features.novelty,
            features.commitment_top_similarity,
            features.tool_pending_pressure,
            1.0f - features.tool_readiness_score,
            features.memory_handle_similarity_variance,
            analytic_score,
            features.social_trust,
        }};
        const float builtin_score = linear_probe_score(-1.30f, weights, probe_features);
        analytic_score = clamp_unit(0.55f * analytic_score + 0.45f * builtin_score);
    }

    if (!params.enable_learned_uncertainty_head || !params.uncertainty_head_callback) {
        return analytic_score;
    }

    float head_score = analytic_score;
    if (!params.uncertainty_head_callback(&features, &head_score, params.uncertainty_head_user_data) || !std::isfinite(head_score)) {
        return analytic_score;
    }

    return clamp_unit(head_score);
}

float llama_self_state::run_broadcast_head(float analytic_score, const llama_self_state_feature_vector & features) const {
    if (params.enable_builtin_broadcast_probe) {
        static constexpr std::array<float, 12> weights = {{
            1.40f, -1.30f, 0.85f, 0.55f, 0.45f, 0.30f,
            0.40f, 0.20f, 0.18f, 0.12f, 0.10f, 0.35f,
        }};
        const std::array<float, 12> probe_features = {{
            analytic_score,
            features.broadcast_inhibition_hint,
            features.followup_hint,
            features.tool_pending_pressure,
            1.0f - features.tool_readiness_score,
            features.goal_top_similarity,
            features.social_bond_strength,
            features.contradiction_score,
            features.uncertainty_score,
            features.memory_handle_top_similarity,
            features.recency_user,
            features.recency_emit,
        }};
        const float builtin_score = linear_probe_score(-0.95f, weights, probe_features);
        analytic_score = clamp_unit(0.60f * analytic_score + 0.40f * builtin_score);
    }

    if (!params.enable_learned_broadcast_head || !params.broadcast_head_callback) {
        return analytic_score;
    }

    float head_score = analytic_score;
    if (!params.broadcast_head_callback(&features, &head_score, params.broadcast_head_user_data) || !std::isfinite(head_score)) {
        return analytic_score;
    }

    return clamp_unit(head_score);
}

bool llama_self_state::build_features(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        bool postwrite,
        llama_self_state_feature_vector * out_features) const {
    if (!vocab || !out_features) {
        return false;
    }

    static const char * const negation_terms[] = {"not", "no", "never", "n't", "cannot", "can't", "failed", "error", "wrong", "false"};
    static const char * const uncertainty_terms[] = {"maybe", "perhaps", "uncertain", "unknown", "unsure", "likely", "possibly"};
    static const char * const self_terms[] = {"i", "me", "my", "myself", "vicuna"};
    static const char * const error_terms[] = {"error", "fail", "failed", "cannot", "can't", "invalid", "denied", "timeout"};
    static const char * const negative_valence_terms[] = {
            "bad", "worse", "wrong", "frustrat", "annoy", "disappoint", "hate", "awful", "terrible", "useless", "upset"
    };

    const auto sketch = build_event_sketch(event);
    const float previous_similarity = has_previous_event_sketch ?
            clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, previous_event_sketch))) : 0.0f;
    float working_memory_top_similarity = 0.0f;
    float working_memory_similarity_variance = 0.0f;
    working_memory_stats(sketch, &working_memory_top_similarity, &working_memory_similarity_variance);
    float memory_handle_top_similarity = 0.0f;
    float memory_handle_similarity_variance = 0.0f;
    memory_handle_stats(sketch, &memory_handle_top_similarity, &memory_handle_similarity_variance);
    const float goal_top_similarity = max_similarity(goals, sketch, false);
    const float commitment_top_similarity = max_similarity(commitments, sketch, true);
    const float identity_similarity = has_identity_sketch ?
            clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, identity_sketch))) : 0.0f;
    const llama_self_tool_state_info tool_state = build_tool_state_info(tool_jobs);

    float negation_hits = 0.0f;
    float uncertainty_hits = 0.0f;
    float self_hits = 0.0f;
    float error_hits = 0.0f;
    float negative_valence_hits = 0.0f;
    uint32_t question_hits = 0;
    std::unordered_set<llama_token> unique_tokens;

    for (size_t i = 0; i < event.n_tokens; ++i) {
        unique_tokens.insert(event.tokens[i]);
        const std::string piece = normalize_piece(vocab->token_to_piece(event.tokens[i]));

        if (contains_any(piece, negation_terms, sizeof(negation_terms)/sizeof(negation_terms[0]))) {
            negation_hits += 1.0f;
        }
        if (contains_any(piece, uncertainty_terms, sizeof(uncertainty_terms)/sizeof(uncertainty_terms[0]))) {
            uncertainty_hits += 1.0f;
        }
        if (contains_any(piece, self_terms, sizeof(self_terms)/sizeof(self_terms[0]))) {
            self_hits += 1.0f;
        }
        if (contains_any(piece, error_terms, sizeof(error_terms)/sizeof(error_terms[0]))) {
            error_hits += 1.0f;
        }
        if (contains_any(piece, negative_valence_terms, sizeof(negative_valence_terms)/sizeof(negative_valence_terms[0]))) {
            negative_valence_hits += 1.0f;
        }
        if (piece.find('?') != std::string::npos) {
            ++question_hits;
        }
    }

    const float token_count = (float) event.n_tokens;
    const float inv_tokens = token_count > 0.0f ? 1.0f / token_count : 0.0f;
    const float unique_ratio = token_count > 0.0f ? (float) unique_tokens.size() / token_count : 0.0f;

    const float continuity_similarity = std::max(previous_similarity, working_memory_top_similarity);
    const float novelty = token_count > 0.0f ? clamp_unit(1.0f - continuity_similarity) : 0.0f;
    const float topic_shift = clamp_unit(0.65f * novelty + 0.35f * (1.0f - previous_similarity));

    const float entropy_feature = event.decoder_entropy > 0.0f ? clamp_unit(event.decoder_entropy / 5.0f) : 0.0f;
    const float top_margin_feature = event.decoder_top_margin > 0.0f ? clamp_unit(event.decoder_top_margin) : 0.0f;

    const float negation_ratio = negation_hits * inv_tokens;
    const float uncertainty_ratio = uncertainty_hits * inv_tokens;
    const float self_ratio = self_hits * inv_tokens;
    const float error_ratio = error_hits * inv_tokens;
    const float negative_valence_ratio = negative_valence_hits * inv_tokens;

    const float contradiction_analytic = clamp_unit(
            0.35f * negation_ratio +
            0.20f * error_ratio +
            0.20f * commitment_top_similarity * std::max(negation_ratio, error_ratio) +
            0.15f * ((event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? 1.0f : 0.0f) +
            0.10f * novelty);

    const float uncertainty_analytic = clamp_unit(std::max(
            uncertainty_ratio,
            0.60f * entropy_feature + 0.40f * (1.0f - top_margin_feature)));
    const float social_bond_strength = clamp_unit(
            0.40f * social_familiarity +
            0.35f * social_trust +
            0.25f * social_reciprocity);
    const bool counterfactual_channel = event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL;

    llama_self_state_feature_vector features = {};
    features.token_count_log = clamp_unit(std::log1pf(token_count) / 8.0f);
    features.unique_token_ratio = clamp_unit(unique_ratio);
    features.novelty = novelty;
    features.topic_shift = topic_shift;
    features.working_memory_top_similarity = working_memory_top_similarity;
    features.working_memory_similarity_variance = working_memory_similarity_variance;
    features.memory_handle_top_similarity = memory_handle_top_similarity;
    features.memory_handle_similarity_variance = memory_handle_similarity_variance;
    features.goal_top_similarity = goal_top_similarity;
    features.commitment_top_similarity = commitment_top_similarity;
    features.identity_similarity = identity_similarity;
    features.self_reference_ratio = clamp_unit(self_ratio);
    features.negation_ratio = clamp_unit(negation_ratio);
    features.uncertainty_lexical_ratio = clamp_unit(uncertainty_ratio);
    features.error_ratio = clamp_unit(error_ratio);
    features.recency_user = decay_to_unit(datetime.delta_since_last_user_ms, params.tool_salience_half_life_ms);
    features.recency_tool = decay_to_unit(datetime.delta_since_last_tool_event_ms, params.tool_salience_half_life_ms);
    features.recency_emit = decay_to_unit(datetime.delta_since_last_emit_ms, params.tool_salience_half_life_ms);
    features.social_familiarity = social_familiarity;
    features.social_trust = social_trust;
    features.social_reciprocity = social_reciprocity;
    features.social_bond_strength = social_bond_strength;
    features.tool_readiness_score = tool_state.readiness;
    features.tool_pending_pressure = clamp_unit(
            0.35f * std::min(1.0f, (float) tool_state.pending_jobs) +
            0.45f * std::min(1.0f, (float) tool_state.running_jobs) +
            0.20f * std::min(1.0f, (float) tool_state.failed_jobs));
    features.decoder_entropy = entropy_feature;
    features.decoder_top_margin = top_margin_feature;
    features.contradiction_score = run_contradiction_head(contradiction_analytic, features);
    features.uncertainty_score = run_uncertainty_head(uncertainty_analytic, features);
    features.memory_write_pressure = clamp_unit(
            updater_program.memory_novelty_weight * features.novelty +
            updater_program.memory_working_similarity_weight * (1.0f - features.working_memory_top_similarity) +
            updater_program.memory_handle_similarity_weight * (1.0f - features.memory_handle_top_similarity) +
            updater_program.memory_uncertainty_weight * features.uncertainty_score +
            updater_program.memory_contradiction_weight * features.contradiction_score +
            updater_program.memory_handle_variance_weight * features.memory_handle_similarity_variance);

    const float social_hint = clamp_unit(
            (event.role == LLAMA_SELF_STATE_EVENT_USER ? 0.55f : 0.20f) +
            0.25f * social_bond_strength +
            0.20f * social_reciprocity);
    const float question_hint = token_count > 0.0f ? clamp_unit((float) question_hits / token_count) : 0.0f;
    const float broadcast_pressure_base = postwrite ? clamp_unit(
            updater_program.broadcast_social_weight * social_hint +
            updater_program.broadcast_contradiction_weight * features.contradiction_score +
            updater_program.broadcast_uncertainty_weight * features.uncertainty_score +
            updater_program.broadcast_tool_pending_weight * features.tool_pending_pressure +
            updater_program.broadcast_tool_unready_weight * (1.0f - features.tool_readiness_score) +
            updater_program.broadcast_failure_weight * ((event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? 1.0f : 0.0f) +
            updater_program.broadcast_question_weight * question_hint +
            updater_program.broadcast_goal_weight * features.goal_top_similarity) : 0.0f;
    features.broadcast_pressure_hint = counterfactual_channel ? 0.15f * broadcast_pressure_base : broadcast_pressure_base;

    const float interruption_guard = channel_state == LLAMA_SELF_STATE_CHANNEL_DO_NOT_INTERRUPT ? 1.0f :
            channel_state == LLAMA_SELF_STATE_CHANNEL_WAITING ? 0.55f : 0.20f;
    const float broadcast_inhibition_base = postwrite ? clamp_unit(
            0.60f * interruption_guard +
            0.20f * (1.0f - features.broadcast_pressure_hint) +
            0.10f * features.tool_pending_pressure +
            0.20f * (event.role == LLAMA_SELF_STATE_EVENT_SYSTEM ? 0.5f : 0.0f)) : 0.0f;
    features.broadcast_inhibition_hint = counterfactual_channel ?
            clamp_unit(std::max(0.80f, broadcast_inhibition_base)) :
            broadcast_inhibition_base;

    features.followup_hint = postwrite ? clamp_unit(
            0.45f * ((event.flags & LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP) ? 1.0f : 0.0f) +
            0.30f * ((event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? 1.0f : 0.0f) +
            0.15f * features.contradiction_score +
            0.10f * features.uncertainty_score) : 0.0f;
    features.negative_user_valence = clamp_unit(negative_valence_ratio);
    if (postwrite) {
        features.broadcast_pressure_hint = run_broadcast_head(features.broadcast_pressure_hint, features);
    }

    *out_features = features;
    return true;
}

bool llama_self_state::build_prewrite_features(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        llama_self_state_feature_vector * out_features) const {
    return build_features(vocab, event, false, out_features);
}

bool llama_self_state::apply_prewrite(const llama_self_state_event & event, const llama_self_state_feature_vector & features) {
    if (!ensure_time_initialized()) {
        return false;
    }

    const uint32_t source_mask = source_mask_for_role(event);
    return apply_register_update_rules(LLAMA_SELF_UPDATER_PHASE_PREWRITE, event, features, source_mask);
}

bool llama_self_state::build_postwrite_features(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        llama_self_state_feature_vector * out_features) const {
    return build_features(vocab, event, true, out_features);
}

bool llama_self_state::apply_postwrite(const llama_self_state_event & event, const llama_self_state_feature_vector & features) {
    if (!ensure_time_initialized()) {
        return false;
    }

    const uint32_t source_mask = source_mask_for_role(event);
    (void) apply_register_update_rules(LLAMA_SELF_UPDATER_PHASE_POSTWRITE, event, features, source_mask);

    update_reactivation_priorities(build_event_sketch(event), features.memory_write_pressure);

    if ((event.flags & LLAMA_SELF_STATE_EVENT_ADMITTED) != 0) {
        admit_working_memory(event, build_event_sketch(event), features);

        if (event.channel != LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) {
            if (event.role == LLAMA_SELF_STATE_EVENT_USER) {
                (void) note_user_event();
                (void) set_channel_state(LLAMA_SELF_STATE_CHANNEL_ACTIVE);
            } else if (event.role == LLAMA_SELF_STATE_EVENT_TOOL ||
                       (event.flags & (LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED | LLAMA_SELF_STATE_EVENT_TOOL_FAILED)) != 0) {
                (void) note_tool_event();
            } else if (event.role == LLAMA_SELF_STATE_EVENT_SYSTEM || (event.flags & LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP) != 0) {
                (void) note_emit_event();
            }
        }
    }

    if (event.channel != LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) {
        update_social_state(event, features);
    }
    previous_event_sketch = build_event_sketch(event);
    has_previous_event_sketch = event.n_tokens > 0;
    append_trace(event);
    return true;
}

void llama_self_state::update_reactivation_priorities(
        const std::array<float, 32> & sketch,
        float memory_write_pressure) {
    reactivation_priorities.clear();

    for (const auto & handle : memory_handles) {
        const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, handle.centroid)));
        const float recency = decay_to_unit(elapsed_or_unset(datetime.monotonic_ms, handle.last_update_monotonic_ms), params.tool_salience_half_life_ms);
        const float priority = clamp_unit(
                0.50f * clamp_unit(memory_write_pressure * similarity) +
                0.30f * handle.priority +
                0.20f * recency);

        if (priority <= 0.0f) {
            continue;
        }

        reactivation_priorities.push_back({
            /*.handle_id =*/ handle.handle_id,
            /*.kind =*/ handle.kind,
            /*.priority =*/ priority,
            /*.top_similarity =*/ similarity,
            /*.last_update_monotonic_ms =*/ datetime.monotonic_ms,
        });
    }

    std::sort(reactivation_priorities.begin(), reactivation_priorities.end(), [](const llama_self_reactivation_info & lhs, const llama_self_reactivation_info & rhs) {
        if (lhs.priority == rhs.priority) {
            return lhs.handle_id < rhs.handle_id;
        }
        return lhs.priority > rhs.priority;
    });
}

void llama_self_state::refresh_tool_surface(uint32_t source_mask) {
    const llama_self_tool_state_info info = build_tool_state_info(tool_jobs);
    const float lifecycle_salience =
            info.failed_jobs > 0 ? 1.0f :
            info.running_jobs > 0 ? 0.92f :
            info.pending_jobs > 0 ? 0.72f :
            info.completed_jobs > 0 ? 0.38f :
            decay_to_unit(datetime.delta_since_last_tool_event_ms, params.tool_salience_half_life_ms);

    update_scalar_register(LLAMA_SELF_REGISTER_TOOL_SALIENCE, lifecycle_salience, source_mask);
}

void llama_self_state::bridge_working_memory_to_handles(
        const std::array<float, 32> & sketch,
        float salience) {
    auto best_it = memory_handles.end();
    float best_similarity = 0.0f;

    for (auto it = memory_handles.begin(); it != memory_handles.end(); ++it) {
        if (it->kind == LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER) {
            const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, it->centroid)));
            if (best_it == memory_handles.end() || similarity > best_similarity) {
                best_it = it;
                best_similarity = similarity;
            }
        }
    }

    if (best_it == memory_handles.end()) {
        return;
    }

    auto & target = *best_it;
    const float blend = clamp_unit(0.20f + 0.60f * salience);
    const float existing_weight = (float) std::max<uint32_t>(1, target.member_count);
    for (size_t i = 0; i < target.centroid.size(); ++i) {
        target.centroid[i] = (1.0f - blend) * target.centroid[i] + blend * sketch[i];
    }

    float norm = 0.0f;
    for (float value : target.centroid) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float & value : target.centroid) {
            value /= norm;
        }
    }

    target.member_count = (uint32_t) std::min<float>(65535.0f, existing_weight + 1.0f);
    target.priority = clamp_unit(0.70f * target.priority + 0.30f * salience);
    target.last_update_monotonic_ms = datetime.monotonic_ms;
}

void llama_self_state::update_social_state(
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features) {
    social_last_update_monotonic_ms = datetime.monotonic_ms;

    if (event.role == LLAMA_SELF_STATE_EVENT_USER) {
        ++social_user_turn_count;
        social_familiarity = clamp_unit(social_familiarity + 0.18f * (1.0f - social_familiarity));
        social_recent_user_valence = clamp_unit(
                0.65f * social_recent_user_valence +
                0.35f * features.negative_user_valence);

        const float response_bonus = clamp_unit(0.45f * features.recency_emit + 0.55f * features.working_memory_top_similarity);
        social_reciprocity = clamp_unit(social_reciprocity + 0.15f * (response_bonus - social_reciprocity));

        const float trust_target = clamp_unit(
                0.55f +
                0.20f * (1.0f - features.contradiction_score) +
                0.15f * (1.0f - features.uncertainty_score) +
                0.10f * features.goal_top_similarity);
        social_trust = clamp_unit(social_trust + 0.10f * (trust_target - social_trust));
    } else {
        ++social_system_turn_count;
        const bool failed = (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) != 0;
        const float trust_target = failed ?
                clamp_unit(0.20f * (1.0f - features.contradiction_score)) :
                clamp_unit(0.45f + 0.25f * (1.0f - features.uncertainty_score) + 0.15f * features.social_bond_strength);
        social_trust = clamp_unit(social_trust + (failed ? 0.18f : 0.08f) * (trust_target - social_trust));

        const float reciprocity_target = clamp_unit(
                0.35f +
                0.25f * features.followup_hint +
                0.20f * features.recency_user +
                0.20f * features.goal_top_similarity);
        social_reciprocity = clamp_unit(social_reciprocity + 0.08f * (reciprocity_target - social_reciprocity));
    }

    social_dissatisfaction = clamp_unit(
            0.50f * social_recent_user_valence +
            0.30f * (1.0f - social_trust) +
            0.20f * (1.0f - social_reciprocity));
}

bool llama_context::self_state_refresh_time() {
    return self_state && self_state->refresh_time();
}

bool llama_context::self_state_set_time(const llama_self_state_time_point & time_point) {
    return self_state && self_state->set_time(time_point);
}

bool llama_context::self_state_get_datetime(llama_self_state_datetime * out_info) const {
    return self_state && self_state->get_datetime(out_info);
}

bool llama_context::self_state_configure(const llama_self_state_params & params) {
    return self_state && self_state->configure(params);
}

int32_t llama_context::self_state_register_count() const {
    return self_state ? self_state->register_count() : 0;
}

bool llama_context::self_state_get_register(int32_t register_id, llama_self_register_info * out_info) const {
    return self_state && self_state->get_register(register_id, out_info);
}

bool llama_context::self_state_set_channel_state(int32_t next_channel_state) {
    return self_state && self_state->set_channel_state(next_channel_state);
}

bool llama_context::self_state_note_user_event() {
    return self_state && self_state->note_user_event();
}

bool llama_context::self_state_note_tool_event() {
    return self_state && self_state->note_tool_event();
}

bool llama_context::self_state_note_emit_event() {
    return self_state && self_state->note_emit_event();
}

bool llama_context::self_state_set_identity(const llama_token * tokens, size_t n_tokens) {
    return self_state && self_state->set_identity(tokens, n_tokens);
}

bool llama_context::self_state_upsert_goal(
        int32_t goal_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    return self_state && self_state->upsert_goal(goal_id, tokens, n_tokens, priority);
}

bool llama_context::self_state_upsert_commitment(
        int32_t commitment_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority,
        bool unresolved) {
    return self_state && self_state->upsert_commitment(commitment_id, tokens, n_tokens, priority, unresolved);
}

int32_t llama_context::self_state_goal_count() const {
    return self_state ? self_state->goal_count() : 0;
}

int32_t llama_context::self_state_commitment_count() const {
    return self_state ? self_state->commitment_count() : 0;
}

int32_t llama_context::self_state_working_memory_count() const {
    return self_state ? self_state->working_memory_count() : 0;
}

bool llama_context::self_state_upsert_memory_handle(
        int32_t handle_id,
        int32_t kind,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    return self_state && self_state->upsert_memory_handle(handle_id, kind, tokens, n_tokens, priority);
}

int32_t llama_context::self_state_memory_handle_count() const {
    return self_state ? self_state->memory_handle_count() : 0;
}

int32_t llama_context::self_state_reactivation_count() const {
    return self_state ? self_state->reactivation_count() : 0;
}

bool llama_context::self_state_get_reactivation(int32_t index, llama_self_reactivation_info * out_info) const {
    return self_state && self_state->get_reactivation(index, out_info);
}

bool llama_context::self_state_upsert_tool_job(int32_t job_id, int32_t status, float importance) {
    return self_state && self_state->upsert_tool_job(job_id, status, importance);
}

bool llama_context::self_state_get_tool_state(llama_self_tool_state_info * out_info) const {
    return self_state && self_state->get_tool_state(out_info);
}

bool llama_context::self_state_get_social_state(llama_self_social_state_info * out_info) const {
    return self_state && self_state->get_social_state(out_info);
}

int32_t llama_context::self_state_trace_count() const {
    return self_state ? self_state->trace_count() : 0;
}

bool llama_context::self_state_clear_trace() {
    return self_state && self_state->clear_trace();
}

bool llama_context::self_state_replay_trace(int32_t upto_count) {
    return self_state && self_state->replay_trace(&get_model().vocab, upto_count, -1);
}

bool llama_context::self_state_replay_trace_on_channel(int32_t upto_count, int32_t replay_channel) {
    return self_state && self_state->replay_trace(&get_model().vocab, upto_count, replay_channel);
}

bool llama_context::self_state_set_updater_program(const llama_self_updater_program & program) {
    return self_state && self_state->set_updater_program(program);
}

bool llama_context::self_state_get_updater_program(llama_self_updater_program * out_program) const {
    return self_state && self_state->get_updater_program(out_program);
}

size_t llama_context::self_state_trace_export_size() const {
    return self_state ? self_state->trace_export_size() : 0;
}

bool llama_context::self_state_trace_export(void * dst, size_t size) const {
    return self_state && self_state->trace_export(dst, size);
}

bool llama_context::self_state_trace_import(const void * src, size_t size, bool replace_existing) {
    return self_state && self_state->trace_import(src, size, replace_existing);
}

bool llama_context::self_state_evaluate_counterfactual(
        const llama_self_updater_program & program,
        int32_t upto_count,
        llama_self_counterfactual_result * out_result) const {
    return self_state && self_state->evaluate_counterfactual(
            &get_model().vocab,
            program,
            upto_count,
            LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
            out_result);
}

bool llama_context::self_state_evaluate_counterfactual_on_channel(
        const llama_self_updater_program & program,
        int32_t upto_count,
        int32_t replay_channel,
        llama_self_counterfactual_result * out_result) const {
    return self_state && self_state->evaluate_counterfactual(&get_model().vocab, program, upto_count, replay_channel, out_result);
}

bool llama_context::self_state_build_prewrite_features(
        const llama_self_state_event & event,
        llama_self_state_feature_vector * out_features) const {
    return self_state && self_state->build_prewrite_features(&get_model().vocab, event, out_features);
}

bool llama_context::self_state_apply_prewrite(
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features) {
    return self_state && self_state->apply_prewrite(event, features);
}

bool llama_context::self_state_build_postwrite_features(
        const llama_self_state_event & event,
        llama_self_state_feature_vector * out_features) const {
    return self_state && self_state->build_postwrite_features(&get_model().vocab, event, out_features);
}

bool llama_context::self_state_apply_postwrite(
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features) {
    if (!self_state) {
        return false;
    }

    const std::vector<float> before = capture_self_register_scalars(*this);
    if (!self_state->apply_postwrite(event, features)) {
        return false;
    }

    if (!hard_memory) {
        return true;
    }

    llama_hard_memory_config config = {};
    if (!hard_memory->get_config(&config) || !config.enabled || !config.archive_enabled) {
        return true;
    }
    if (event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL && !config.archive_counterfactual_events) {
        return true;
    }
    if (!event.tokens || event.n_tokens == 0) {
        return true;
    }

    const std::vector<float> after = capture_self_register_scalars(*this);
    const llama_self_state_delta_summary delta = summarize_self_state_delta(before, after, event);
    if (delta.total_delta < config.archival_delta_threshold) {
        return true;
    }

    (void) hard_memory->archive_event(&get_model().vocab, event, delta);
    return true;
}

bool llama_context::hard_memory_configure(const llama_hard_memory_config & config) {
    return hard_memory && hard_memory->configure(config);
}

bool llama_context::hard_memory_get_config(llama_hard_memory_config * out_config) const {
    return hard_memory && hard_memory->get_config(out_config);
}

bool llama_context::hard_memory_query(const llama_hard_memory_query_request & query, llama_hard_memory_result * out_result) {
    return hard_memory && hard_memory->query(query, out_result);
}

bool llama_context::hard_memory_get_last_result(llama_hard_memory_result * out_result) const {
    return hard_memory && hard_memory->get_last_result(out_result);
}

bool llama_context::hard_memory_get_last_archive_trace(llama_hard_memory_archive_trace * out_trace) const {
    return hard_memory && hard_memory->get_last_archive_trace(out_trace);
}

llama_self_state_params llama_self_state_default_params(void) {
    return {
        /*.enable_learned_contradiction_head =*/ false,
        /*.enable_learned_uncertainty_head =*/ false,
        /*.enable_learned_broadcast_head =*/ false,
        /*.enable_builtin_contradiction_probe =*/ true,
        /*.enable_builtin_uncertainty_probe =*/ true,
        /*.enable_builtin_broadcast_probe =*/ true,
        /*.tool_salience_half_life_ms =*/ LLAMA_SELF_DEFAULT_TOOL_SALIENCE_HALF_LIFE_MS,
        /*.prewrite_gain =*/ 0.65f,
        /*.postwrite_gain =*/ 0.55f,
        /*.contradiction_head_callback =*/ nullptr,
        /*.contradiction_head_user_data =*/ nullptr,
        /*.uncertainty_head_callback =*/ nullptr,
        /*.uncertainty_head_user_data =*/ nullptr,
        /*.broadcast_head_callback =*/ nullptr,
        /*.broadcast_head_user_data =*/ nullptr,
    };
}

llama_self_updater_program llama_self_state_default_updater_program(void) {
    llama_self_updater_program program = {};
    program.version = 1;
    program.memory_novelty_weight = 0.26f;
    program.memory_working_similarity_weight = 0.14f;
    program.memory_handle_similarity_weight = 0.10f;
    program.memory_uncertainty_weight = 0.25f;
    program.memory_contradiction_weight = 0.20f;
    program.memory_handle_variance_weight = 0.05f;
    program.broadcast_social_weight = 0.24f;
    program.broadcast_contradiction_weight = 0.18f;
    program.broadcast_uncertainty_weight = 0.18f;
    program.broadcast_tool_pending_weight = 0.14f;
    program.broadcast_tool_unready_weight = 0.08f;
    program.broadcast_failure_weight = 0.15f;
    program.broadcast_question_weight = 0.08f;
    program.broadcast_goal_weight = 0.13f;
    program.repair_emit_threshold = 0.72f;
    program.repair_dissatisfaction_floor = 0.38f;
    program.repair_recent_user_valence_floor = 0.20f;
    program.repair_inhibition_max = 0.68f;
    program.repair_admission_floor = 0.34f;
    program.repair_admission_weight = 0.22f;

    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_UNCERTAINTY,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.08f, 0.58f, 0.42f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY, 0.72f},
                {LLAMA_SELF_UPDATER_FEATURE_DECODER_ENTROPY, 0.18f},
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_FAILED, 0.08f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_CONTRADICTION,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.05f, 0.60f, 0.40f, 0.02f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION, 0.74f},
                {LLAMA_SELF_UPDATER_FEATURE_NEGATION_RATIO, 0.10f},
                {LLAMA_SELF_UPDATER_FEATURE_ERROR_RATIO, 0.10f},
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_FAILED, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_NOVELTY,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.04f, 0.72f, 0.46f, 0.04f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_NOVELTY, 0.82f},
                {LLAMA_SELF_UPDATER_FEATURE_TOPIC_SHIFT, 0.10f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_TOPIC_SHIFT,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.04f, 0.70f, 0.44f, 0.04f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_TOPIC_SHIFT, 0.86f},
                {LLAMA_SELF_UPDATER_FEATURE_NOVELTY, 0.08f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_GOAL_RELEVANCE,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.05f, 0.62f, 0.36f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.78f},
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_USER, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_SELF_RELEVANCE,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.04f, 0.60f, 0.34f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_SELF_REFERENCE, 0.34f},
                {LLAMA_SELF_UPDATER_FEATURE_IDENTITY_SIMILARITY, 0.34f},
                {LLAMA_SELF_UPDATER_FEATURE_COMMITMENT_SIMILARITY, 0.22f},
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_SOCIAL_RELEVANCE,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.08f, 0.58f, 0.32f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_USER, 0.24f},
                {LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND, 0.42f},
                {LLAMA_SELF_UPDATER_FEATURE_SOCIAL_RECIPROCITY, 0.14f},
                {LLAMA_SELF_UPDATER_FEATURE_RECENCY_USER, 0.10f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_AFFORDANCE,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.05f, 0.62f, 0.38f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_ERROR_RATIO, 0.34f},
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.26f},
                {LLAMA_SELF_UPDATER_FEATURE_TOOL_PENDING_PRESSURE, 0.14f},
                {LLAMA_SELF_UPDATER_FEATURE_TOOL_READINESS, -0.10f},
                {LLAMA_SELF_UPDATER_FEATURE_SELF_REFERENCE, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.05f, 0.64f, 0.40f, 0.02f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_MEMORY_WRITE_PRESSURE, 0.82f},
                {LLAMA_SELF_UPDATER_FEATURE_NOVELTY, 0.06f},
                {LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY, 0.04f},
                {LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION, 0.04f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_BROADCAST_PRESSURE,
            LLAMA_SELF_UPDATER_PHASE_POSTWRITE,
            0.03f, 0.58f, 0.34f, 0.02f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_BROADCAST_PRESSURE_HINT, 0.74f},
                {LLAMA_SELF_UPDATER_FEATURE_FOLLOWUP_HINT, 0.10f},
                {LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_BROADCAST_INHIBITION,
            LLAMA_SELF_UPDATER_PHASE_POSTWRITE,
            0.06f, 0.60f, 0.36f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_BROADCAST_INHIBITION_HINT, 0.82f},
                {LLAMA_SELF_UPDATER_FEATURE_TOOL_PENDING_PRESSURE, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY, 0.06f},
            },
            {
                {LLAMA_SELF_REGISTER_BROADCAST_PRESSURE, -0.08f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION,
            LLAMA_SELF_UPDATER_PHASE_POSTWRITE,
            0.04f, 0.62f, 0.38f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_FOLLOWUP_HINT, 0.78f},
                {LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.04f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY,
            LLAMA_SELF_UPDATER_PHASE_POSTWRITE,
            0.05f, 0.60f, 0.38f, 0.02f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_MEMORY_WRITE_PRESSURE, 0.78f},
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_ADMITTED, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_BROADCAST_PRESSURE_HINT, 0.04f},
            });

    return program;
}

int32_t llama_self_state_refresh_time(struct llama_context * ctx) {
    return ctx && ctx->self_state_refresh_time() ? 0 : -1;
}

int32_t llama_self_state_set_time(
        struct llama_context * ctx,
        struct llama_self_state_time_point time_point) {
    return ctx && ctx->self_state_set_time(time_point) ? 0 : -1;
}

int32_t llama_self_state_get_datetime(
        const struct llama_context * ctx,
        struct llama_self_state_datetime * out_info) {
    return ctx && ctx->self_state_get_datetime(out_info) ? 0 : -1;
}

int32_t llama_self_state_register_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_register_count() : -1;
}

int32_t llama_self_state_get_register(
        const struct llama_context * ctx,
        int32_t register_id,
        struct llama_self_register_info * out_info) {
    return ctx && ctx->self_state_get_register(register_id, out_info) ? 0 : -1;
}

const char * llama_self_state_register_name(int32_t register_id) {
    return llama_self_state::register_name(register_id);
}

int32_t llama_self_state_configure(
        struct llama_context * ctx,
        struct llama_self_state_params params) {
    return ctx && ctx->self_state_configure(params) ? 0 : -1;
}

int32_t llama_self_state_set_channel_state(
        struct llama_context * ctx,
        int32_t channel_state) {
    return ctx && ctx->self_state_set_channel_state(channel_state) ? 0 : -1;
}

int32_t llama_self_state_note_user_event(struct llama_context * ctx) {
    return ctx && ctx->self_state_note_user_event() ? 0 : -1;
}

int32_t llama_self_state_note_tool_event(struct llama_context * ctx) {
    return ctx && ctx->self_state_note_tool_event() ? 0 : -1;
}

int32_t llama_self_state_note_emit_event(struct llama_context * ctx) {
    return ctx && ctx->self_state_note_emit_event() ? 0 : -1;
}

int32_t llama_self_state_set_identity(
        struct llama_context * ctx,
        const llama_token * tokens,
        size_t n_tokens) {
    return ctx && ctx->self_state_set_identity(tokens, n_tokens) ? 0 : -1;
}

int32_t llama_self_state_upsert_goal(
        struct llama_context * ctx,
        int32_t goal_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    return ctx && ctx->self_state_upsert_goal(goal_id, tokens, n_tokens, priority) ? 0 : -1;
}

int32_t llama_self_state_upsert_commitment(
        struct llama_context * ctx,
        int32_t commitment_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority,
        bool unresolved) {
    return ctx && ctx->self_state_upsert_commitment(commitment_id, tokens, n_tokens, priority, unresolved) ? 0 : -1;
}

int32_t llama_self_state_goal_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_goal_count() : -1;
}

int32_t llama_self_state_commitment_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_commitment_count() : -1;
}

int32_t llama_self_state_working_memory_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_working_memory_count() : -1;
}

int32_t llama_self_state_upsert_memory_handle(
        struct llama_context * ctx,
        int32_t handle_id,
        int32_t kind,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    return ctx && ctx->self_state_upsert_memory_handle(handle_id, kind, tokens, n_tokens, priority) ? 0 : -1;
}

int32_t llama_self_state_memory_handle_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_memory_handle_count() : -1;
}

int32_t llama_self_state_reactivation_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_reactivation_count() : -1;
}

int32_t llama_self_state_get_reactivation(
        const struct llama_context * ctx,
        int32_t index,
        struct llama_self_reactivation_info * out_info) {
    return ctx && ctx->self_state_get_reactivation(index, out_info) ? 0 : -1;
}

int32_t llama_self_state_upsert_tool_job(
        struct llama_context * ctx,
        int32_t job_id,
        int32_t status,
        float importance) {
    return ctx && ctx->self_state_upsert_tool_job(job_id, status, importance) ? 0 : -1;
}

int32_t llama_self_state_get_tool_state(
        const struct llama_context * ctx,
        struct llama_self_tool_state_info * out_info) {
    return ctx && ctx->self_state_get_tool_state(out_info) ? 0 : -1;
}

int32_t llama_self_state_get_social_state(
        const struct llama_context * ctx,
        struct llama_self_social_state_info * out_info) {
    return ctx && ctx->self_state_get_social_state(out_info) ? 0 : -1;
}

int32_t llama_self_state_trace_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_trace_count() : -1;
}

int32_t llama_self_state_clear_trace(struct llama_context * ctx) {
    return ctx && ctx->self_state_clear_trace() ? 0 : -1;
}

int32_t llama_self_state_replay_trace(struct llama_context * ctx, int32_t upto_count) {
    return ctx && ctx->self_state_replay_trace(upto_count) ? 0 : -1;
}

int32_t llama_self_state_replay_trace_on_channel(
        struct llama_context * ctx,
        int32_t upto_count,
        int32_t replay_channel) {
    return ctx && ctx->self_state_replay_trace_on_channel(upto_count, replay_channel) ? 0 : -1;
}

int32_t llama_self_state_set_updater_program(
        struct llama_context * ctx,
        struct llama_self_updater_program program) {
    return ctx && ctx->self_state_set_updater_program(program) ? 0 : -1;
}

int32_t llama_self_state_get_updater_program(
        const struct llama_context * ctx,
        struct llama_self_updater_program * out_program) {
    return ctx && ctx->self_state_get_updater_program(out_program) ? 0 : -1;
}

size_t llama_self_state_trace_export_size(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_trace_export_size() : 0;
}

int32_t llama_self_state_trace_export(
        const struct llama_context * ctx,
        void * dst,
        size_t size) {
    return ctx && ctx->self_state_trace_export(dst, size) ? 0 : -1;
}

int32_t llama_self_state_trace_import(
        struct llama_context * ctx,
        const void * src,
        size_t size,
        bool replace_existing) {
    return ctx && ctx->self_state_trace_import(src, size, replace_existing) ? 0 : -1;
}

int32_t llama_self_state_evaluate_counterfactual(
        const struct llama_context * ctx,
        struct llama_self_updater_program program,
        int32_t upto_count,
        struct llama_self_counterfactual_result * out_result) {
    return ctx && ctx->self_state_evaluate_counterfactual(program, upto_count, out_result) ? 0 : -1;
}

int32_t llama_self_state_evaluate_counterfactual_on_channel(
        const struct llama_context * ctx,
        struct llama_self_updater_program program,
        int32_t upto_count,
        int32_t replay_channel,
        struct llama_self_counterfactual_result * out_result) {
    return ctx && ctx->self_state_evaluate_counterfactual_on_channel(program, upto_count, replay_channel, out_result) ? 0 : -1;
}

int32_t llama_self_state_build_prewrite_features(
        const struct llama_context * ctx,
        const struct llama_self_state_event * event,
        struct llama_self_state_feature_vector * out_features) {
    return ctx && event && ctx->self_state_build_prewrite_features(*event, out_features) ? 0 : -1;
}

int32_t llama_self_state_apply_prewrite(
        struct llama_context * ctx,
        const struct llama_self_state_event * event,
        const struct llama_self_state_feature_vector * features) {
    return ctx && event && features && ctx->self_state_apply_prewrite(*event, *features) ? 0 : -1;
}

int32_t llama_self_state_build_postwrite_features(
        const struct llama_context * ctx,
        const struct llama_self_state_event * event,
        struct llama_self_state_feature_vector * out_features) {
    return ctx && event && ctx->self_state_build_postwrite_features(*event, out_features) ? 0 : -1;
}

int32_t llama_self_state_apply_postwrite(
        struct llama_context * ctx,
        const struct llama_self_state_event * event,
        const struct llama_self_state_feature_vector * features) {
    return ctx && event && features && ctx->self_state_apply_postwrite(*event, *features) ? 0 : -1;
}

int32_t llama_hard_memory_configure(
        struct llama_context * ctx,
        struct llama_hard_memory_config config) {
    return ctx && ctx->hard_memory_configure(config) ? 0 : -1;
}

int32_t llama_hard_memory_get_config(
        const struct llama_context * ctx,
        struct llama_hard_memory_config * out_config) {
    return ctx && ctx->hard_memory_get_config(out_config) ? 0 : -1;
}

int32_t llama_hard_memory_query(
        struct llama_context * ctx,
        const struct llama_hard_memory_query_request * query,
        struct llama_hard_memory_result * out_result) {
    return ctx && query && out_result && ctx->hard_memory_query(*query, out_result) ? 0 : -1;
}

int32_t llama_hard_memory_get_last_result(
        const struct llama_context * ctx,
        struct llama_hard_memory_result * out_result) {
    return ctx && out_result && ctx->hard_memory_get_last_result(out_result) ? 0 : -1;
}

int32_t llama_hard_memory_get_last_archive_trace(
        const struct llama_context * ctx,
        struct llama_hard_memory_archive_trace * out_trace) {
    return ctx && out_trace && ctx->hard_memory_get_last_archive_trace(out_trace) ? 0 : -1;
}
