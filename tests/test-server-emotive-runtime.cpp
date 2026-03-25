#include "../tools/server/server-emotive-runtime.h"

#include <cmath>
#include <cstdio>

static bool expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        return false;
    }
    return true;
}

static bool approx_zero(float value) {
    return std::fabs(value) < 1e-6f;
}

int main() {
    server_emotive_runtime_config config;
    config.enabled = true;
    config.block_max_chars = 80;
    config.max_blocks_per_turn = 16;
    config.max_turn_history = 2;
    config.embedding.enabled = false;
    config.embedding.model_path.clear();

    server_emotive_runtime runtime(config);

    server_emotive_trace first_trace;
    {
        server_emotive_turn_builder builder(runtime, "deepseek-reasoner");
        builder.add_user_message("Please inspect the logs and explain what failed.");
        builder.observe_reasoning_delta("First I should inspect the failure and check whether there is a contradiction.");
        builder.observe_content_delta("I found the timeout in the provider log and the likely cause is a missing retry.");
        builder.observe_runtime_event("provider_finish:stop");
        first_trace = builder.finalize();
    }

    if (!expect(first_trace.valid, "expected first trace to be valid")) {
        return 1;
    }
    if (!expect(first_trace.blocks.size() >= 4, "expected first trace to retain user, reasoning, assistant, and runtime blocks")) {
        return 1;
    }
    if (!expect(first_trace.blocks.front().kind == SERVER_EMOTIVE_BLOCK_USER_MESSAGE, "expected first block to be a user message")) {
        return 1;
    }
    if (!expect(approx_zero(first_trace.blocks.front().delta.negative_mass), "expected first block to have zero delta baseline")) {
        return 1;
    }
    if (!expect(approx_zero(first_trace.blocks.front().vad.trend.d_valence) &&
                approx_zero(first_trace.blocks.front().vad.trend.d_arousal) &&
                approx_zero(first_trace.blocks.front().vad.trend.d_dominance),
                "expected first block to have zero VAD trend baseline")) {
        return 1;
    }

    bool saw_reasoning = false;
    bool saw_content = false;
    bool saw_runtime = false;
    bool saw_nonzero_delta = false;
    bool saw_nonzero_vad_trend = false;
    for (const server_emotive_block_record & block : first_trace.blocks) {
        saw_reasoning = saw_reasoning || block.kind == SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING;
        saw_content = saw_content || block.kind == SERVER_EMOTIVE_BLOCK_ASSISTANT_CONTENT;
        saw_runtime = saw_runtime || block.kind == SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT;
        saw_nonzero_delta = saw_nonzero_delta || std::fabs(block.delta.d_momentum) > 1e-6f ||
                std::fabs(block.delta.d_confidence) > 1e-6f ||
                std::fabs(block.delta.d_runtime_trust) > 1e-6f;
        saw_nonzero_vad_trend = saw_nonzero_vad_trend ||
                std::fabs(block.vad.trend.d_valence) > 1e-6f ||
                std::fabs(block.vad.trend.d_arousal) > 1e-6f ||
                std::fabs(block.vad.trend.d_dominance) > 1e-6f;
        if (!expect(block.vad.arousal >= 0.0f && block.vad.arousal <= 1.0f, "expected arousal projection to remain bounded")) {
            return 1;
        }
        if (!expect(block.vad.valence >= -1.0f && block.vad.valence <= 1.0f, "expected valence projection to remain bounded")) {
            return 1;
        }
        if (!expect(block.vad.dominance >= -1.0f && block.vad.dominance <= 1.0f, "expected dominance projection to remain bounded")) {
            return 1;
        }
        if (!expect(block.vad.style_guide.warmth >= 0.0f && block.vad.style_guide.warmth <= 1.0f, "expected warmth to remain bounded")) {
            return 1;
        }
        if (!expect(block.vad.style_guide.energy >= 0.0f && block.vad.style_guide.energy <= 1.0f, "expected energy to remain bounded")) {
            return 1;
        }
        if (!expect(block.vad.style_guide.assertiveness >= 0.0f && block.vad.style_guide.assertiveness <= 1.0f, "expected assertiveness to remain bounded")) {
            return 1;
        }
        if (!expect(!block.vad.style_guide.tone_label.empty(), "expected each block to have a tone label")) {
            return 1;
        }
        if (!expect(!block.vad.labels.empty(), "expected each block to include VAD labels")) {
            return 1;
        }
        if (!expect(!block.vad.dominant_dimensions.empty(), "expected each block to include dominant dimensions")) {
            return 1;
        }
        if (!expect(!block.vad.style_guide.prompt_hints.empty(), "expected each block to include prompt hints")) {
            return 1;
        }
    }

    if (!expect(saw_reasoning, "expected trace to capture assistant reasoning blocks")) {
        return 1;
    }
    if (!expect(saw_content, "expected trace to capture assistant content blocks")) {
        return 1;
    }
    if (!expect(saw_runtime, "expected trace to capture runtime event blocks")) {
        return 1;
    }
    if (!expect(saw_nonzero_delta, "expected later blocks to produce non-zero deltas")) {
        return 1;
    }
    if (!expect(saw_nonzero_vad_trend, "expected later blocks to produce non-zero VAD trend")) {
        return 1;
    }

    {
        server_emotive_turn_builder builder(runtime, "deepseek-reasoner");
        builder.add_user_message("Second turn.");
        builder.observe_content_delta("Acknowledged.");
        builder.observe_runtime_event("provider_finish:stop");
        (void) builder.finalize();
    }
    server_emotive_trace third_trace;
    {
        server_emotive_turn_builder builder(runtime, "deepseek-reasoner");
        builder.add_user_message("Third turn.");
        builder.observe_content_delta("Complete.");
        builder.observe_runtime_event("provider_finish:stop");
        third_trace = builder.finalize();
    }

    const json latest = runtime.latest_trace_json();
    if (!expect(latest.at("retained_turns").get<int>() == 2, "expected retained turn history to respect the configured bound")) {
        return 1;
    }
    if (!expect(latest.at("trace").at("trace_id").get<std::string>() == third_trace.trace_id, "expected latest trace surface to expose the most recent trace")) {
        return 1;
    }

    return 0;
}
