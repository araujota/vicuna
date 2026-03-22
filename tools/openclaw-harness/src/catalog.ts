import type { CapabilityCatalog, CapabilityDescriptor } from "./contracts.js";
import { assertCapabilityCatalog } from "./contracts.js";
import type { OpenClawToolSecrets } from "./config.js";

export type BuiltinToolId = "exec" | "hard_memory_query" | "hard_memory_write" | "codex";

export type CatalogOptions = {
  enabledTools?: BuiltinToolId[];
  enableExec?: boolean;
  enableHardMemoryQuery?: boolean;
  enableHardMemoryWrite?: boolean;
  enableCodex?: boolean;
};

export type RuntimeCatalogOptions = {
  secrets?: OpenClawToolSecrets;
};

const COG_TOOL_FLAG_ACTIVE_ELIGIBLE = 1 << 0;
const COG_TOOL_FLAG_DMN_ELIGIBLE = 1 << 1;
const COG_TOOL_FLAG_SIMULATION_SAFE = 1 << 2;
const COG_TOOL_FLAG_REMEDIATION_SAFE = 1 << 3;
const COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT = 1 << 4;

function combineToolFlags(...flags: number[]): number {
  return flags.reduce((mask, flag) => mask | flag, 0);
}

function execCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.exec.command",
    tool_surface_id: "vicuna.exec.main",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-core",
    tool_name: "exec",
    description:
      "Inspect or act on host-local state by running one bounded shell command through the execution policy. Use this for filesystem state, the current working directory, repository state, environment variables, running processes, or command output. Keep the command narrow and direct, and do not use shell chaining or redirection.",
    input_schema_json: {
      type: "object",
      required: ["command"],
      properties: {
        command: {
          type: "string",
          description:
            "A single bounded shell command to execute for direct host-local observation or action. Prefer precise commands such as pwd, ls, git status, or cat path."
        },
        workdir: {
          type: "string",
          description:
            "Optional working directory for the command when the observation or action should run in a specific repository or path."
        }
      }
    },
    output_contract: "pending_then_result",
    side_effect_class: "system_exec",
    approval_mode: "policy_driven",
    execution_modes: ["sync", "background", "approval_gated"],
    provenance_namespace: "openclaw/openclaw-core/tool/exec",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_REMEDIATION_SAFE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_bash"
  };
}

function hardMemoryCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.hard_memory_query",
    tool_surface_id: "vicuna.memory.hard_query",
    capability_kind: "memory_adapter",
    owner_plugin_id: "vicuna-memory",
    tool_name: "hard_memory_query",
    description: "Query Vicuña hard memory with typed results",
    input_schema_json: {
      type: "object",
      required: ["query"],
      properties: {
        query: {
          type: "string",
          description:
            "The retrieval query to run against Vicuña hard memory when looking for relevant durable memories or prior tool observations."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_read",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-memory/memory_adapter/hard_memory_query",
    tool_kind: 2,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_SIMULATION_SAFE,
      COG_TOOL_FLAG_REMEDIATION_SAFE
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_hard_memory"
  };
}

function hardMemoryWriteCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.hard_memory_write",
    tool_surface_id: "vicuna.memory.hard_write",
    capability_kind: "memory_adapter",
    owner_plugin_id: "vicuna-memory",
    tool_name: "hard_memory_write",
    description: "Archive explicit durable memories to Vicuña hard memory and Supermemory",
    input_schema_json: {
      type: "object",
      required: ["memories"],
      properties: {
        memories: {
          type: "array",
          description:
            "The batch of durable memory primitives to archive into Vicuña hard memory and Supermemory.",
          minItems: 1,
          items: {
            type: "object",
            description:
              "One durable memory primitive to write, including its content and optional metadata.",
            required: ["content"],
            properties: {
              content: {
                type: "string",
                description: "The durable memory content to archive."
              },
              title: {
                type: "string",
                description: "An optional short title for the memory."
              },
              key: {
                type: "string",
                description: "An optional stable key used to update the same semantic memory over time."
              },
              kind: {
                type: "string",
                description: "The primitive kind, such as TRAJECTORY, OUTCOME, TOOL_OBSERVATION, USER_MODEL, or SELF_MODEL_FRAGMENT."
              },
              domain: {
                type: "string",
                description: "An optional domain label that scopes the memory."
              },
              tags: {
                type: "array",
                description: "Optional tags that help classify or retrieve the memory later.",
                items: {
                  type: "string",
                  description: "One tag for the memory."
                }
              },
              importance: {
                type: "number",
                description: "Normalized importance weight in the range [0, 1]."
              },
              confidence: {
                type: "number",
                description: "Normalized confidence score in the range [0, 1]."
              },
              gainBias: {
                type: "number",
                description: "Normalized gain-bias contribution for the archived memory."
              },
              allostaticRelevance: {
                type: "number",
                description: "Normalized allostatic relevance score in the range [0, 1]."
              },
              isStatic: {
                type: "boolean",
                description: "Whether this memory should be treated as static rather than updated dynamically."
              }
            }
          }
        },
        containerTag: {
          type: "string",
          description: "An optional container tag used to group this write batch."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_write",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-memory/memory_adapter/hard_memory_write",
    tool_kind: 3,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_REMEDIATION_SAFE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 2,
    max_steps_reserved: 3,
    dispatch_backend: "legacy_hard_memory"
  };
}

function codexCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.codex_cli",
    tool_surface_id: "vicuna.codex.main",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-runtime",
    tool_name: "codex",
    description: "Use the local Codex CLI to implement a repository change and rebuild the runtime",
    input_schema_json: {
      type: "object",
      required: ["task"],
      properties: {
        task: {
          type: "string",
          description:
            "The repository change or runtime task for the local Codex CLI to perform."
        }
      }
    },
    output_contract: "pending_then_result",
    side_effect_class: "self_modification",
    approval_mode: "none",
    execution_modes: ["background"],
    provenance_namespace: "openclaw/vicuna-runtime/tool/codex",
    tool_kind: 5,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_REMEDIATION_SAFE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 2,
    max_steps_reserved: 3,
    dispatch_backend: "legacy_codex"
  };
}

function tavilyWebSearchCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.tavily.web_search",
    tool_surface_id: "vicuna.web.search.tavily",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-tavily",
    tool_name: "web_search",
    description: "Search the live web through Tavily and return ranked source evidence with URLs and excerpts; synthesize from the returned sources rather than expecting a provider-generated answer",
    input_schema_json: {
      type: "object",
      required: ["query"],
      properties: {
        query: {
          type: "string",
          description: "The live web search query to run through Tavily."
        },
        topic: {
          type: "string",
          enum: ["general", "news", "finance"],
          description: "The retrieval topic that best matches the query."
        },
        search_depth: {
          type: "string",
          enum: ["basic", "advanced"],
          description: "How aggressively Tavily should retrieve and expand source evidence."
        },
        max_results: {
          type: "integer",
          minimum: 3,
          maximum: 8,
          default: 5,
          description: "The maximum number of ranked sources to return."
        },
        time_range: {
          type: "string",
          enum: ["day", "week", "month", "year"],
          description: "An optional recency window for the search."
        },
        include_domains: {
          type: "array",
          description: "Optional domains that Tavily should prefer or restrict results to.",
          items: {
            type: "string",
            description: "One domain to include in the search."
          }
        },
        exclude_domains: {
          type: "array",
          description: "Optional domains that Tavily should exclude from the search.",
          items: {
            type: "string",
            description: "One domain to exclude from the search."
          }
        },
        country: {
          type: "string",
          description: "An optional country hint used to localize the search."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "network_read",
    approval_mode: "policy_driven",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/openclaw-tavily/tool/web_search",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_REMEDIATION_SAFE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_bash"
  };
}

function radarrCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.servarr.radarr",
    tool_surface_id: "vicuna.media.radarr",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-servarr",
    tool_name: "radarr",
    description:
      "Inspect and manage the Radarr movie library on the LAN-connected media server. Use this for Radarr system status, queue state, calendar, root folders, quality profiles, existing movies, movie lookup, and adding a movie after you have the right folder and quality profile.",
    input_schema_json: {
      type: "object",
      required: ["action"],
      properties: {
        action: {
          type: "string",
          enum: [
            "system_status",
            "queue",
            "calendar",
            "root_folders",
            "quality_profiles",
            "list_movies",
            "lookup_movie",
            "add_movie",
          ],
          description:
            "The Radarr operation to perform. Read actions inspect the movie library; add_movie looks up a movie and then adds it to Radarr."
        },
        term: {
          type: "string",
          description:
            "Search term for lookup_movie or add_movie. Use a precise movie title, year, or other lookup phrase."
        },
        tmdb_id: {
          type: "integer",
          description:
            "Optional TMDb movie id used to disambiguate the chosen lookup result when add_movie finds multiple candidates."
        },
        start: {
          type: "string",
          description:
            "Inclusive ISO 8601 start timestamp for the calendar action."
        },
        end: {
          type: "string",
          description:
            "Inclusive ISO 8601 end timestamp for the calendar action."
        },
        include_unmonitored: {
          type: "boolean",
          description:
            "Whether the calendar action should include unmonitored movies."
        },
        root_folder_path: {
          type: "string",
          description:
            "Destination Radarr root folder path. Required for add_movie."
        },
        quality_profile_id: {
          type: "integer",
          description:
            "Radarr quality profile id to assign when adding a movie. Required for add_movie."
        },
        monitored: {
          type: "boolean",
          description:
            "Whether the added movie should be monitored in Radarr."
        },
        minimum_availability: {
          type: "string",
          enum: ["tba", "announced", "inCinemas", "released"],
          description:
            "Minimum Radarr availability state to wait for when adding a movie."
        },
        monitor: {
          type: "string",
          enum: ["movieOnly", "movieAndCollection", "none"],
          description:
            "Radarr add-monitor mode used inside add_movie addOptions."
        },
        search_for_movie: {
          type: "boolean",
          description:
            "Whether Radarr should immediately search for the movie after adding it."
        },
        tags: {
          type: "array",
          description:
            "Optional Radarr tag ids to assign to the movie when adding it.",
          items: {
            type: "integer",
            description: "One Radarr tag id."
          }
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "service_api",
    approval_mode: "policy_driven",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/openclaw-servarr/tool/radarr",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 3,
    dispatch_backend: "legacy_bash"
  };
}

function sonarrCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.servarr.sonarr",
    tool_surface_id: "vicuna.media.sonarr",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-servarr",
    tool_name: "sonarr",
    description:
      "Inspect and manage the Sonarr series library on the LAN-connected media server. Use this for Sonarr system status, queue state, calendar, root folders, quality profiles, existing series, series lookup, and adding a series after you have the right folder and quality profile.",
    input_schema_json: {
      type: "object",
      required: ["action"],
      properties: {
        action: {
          type: "string",
          enum: [
            "system_status",
            "queue",
            "calendar",
            "root_folders",
            "quality_profiles",
            "list_series",
            "lookup_series",
            "add_series",
          ],
          description:
            "The Sonarr operation to perform. Read actions inspect the series library; add_series looks up a series and then adds it to Sonarr."
        },
        term: {
          type: "string",
          description:
            "Search term for lookup_series or add_series. Use a precise series title, year, or other lookup phrase."
        },
        tvdb_id: {
          type: "integer",
          description:
            "Optional TVDb series id used to disambiguate the chosen lookup result when add_series finds multiple candidates."
        },
        tmdb_id: {
          type: "integer",
          description:
            "Optional TMDb series id used to disambiguate the chosen lookup result when add_series finds multiple candidates."
        },
        start: {
          type: "string",
          description:
            "Inclusive ISO 8601 start timestamp for the calendar action."
        },
        end: {
          type: "string",
          description:
            "Inclusive ISO 8601 end timestamp for the calendar action."
        },
        include_unmonitored: {
          type: "boolean",
          description:
            "Whether the calendar action should include unmonitored series."
        },
        root_folder_path: {
          type: "string",
          description:
            "Destination Sonarr root folder path. Required for add_series."
        },
        quality_profile_id: {
          type: "integer",
          description:
            "Sonarr quality profile id to assign when adding a series. Required for add_series."
        },
        monitored: {
          type: "boolean",
          description:
            "Whether the added series should be monitored in Sonarr."
        },
        season_folder: {
          type: "boolean",
          description:
            "Whether Sonarr should create season folders for the added series."
        },
        series_type: {
          type: "string",
          enum: ["standard", "daily", "anime"],
          description:
            "Sonarr series type for the added series."
        },
        monitor_new_items: {
          type: "string",
          enum: ["all", "none"],
          description:
            "How Sonarr should monitor newly discovered items for the added series."
        },
        monitor: {
          type: "string",
          enum: [
            "unknown",
            "all",
            "future",
            "missing",
            "existing",
            "firstSeason",
            "lastSeason",
            "latestSeason",
            "pilot",
            "recent",
            "monitorSpecials",
            "unmonitorSpecials",
            "none",
            "skip",
          ],
          description:
            "Sonarr episode-monitor mode used inside add_series addOptions."
        },
        search_for_missing_episodes: {
          type: "boolean",
          description:
            "Whether Sonarr should immediately search for missing episodes after adding the series."
        },
        search_for_cutoff_unmet_episodes: {
          type: "boolean",
          description:
            "Whether Sonarr should immediately search for cutoff-unmet episodes after adding the series."
        },
        tags: {
          type: "array",
          description:
            "Optional Sonarr tag ids to assign to the series when adding it.",
          items: {
            type: "integer",
            description: "One Sonarr tag id."
          }
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "service_api",
    approval_mode: "policy_driven",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/openclaw-servarr/tool/sonarr",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 3,
    dispatch_backend: "legacy_bash"
  };
}

const BUILTIN_CAPABILITIES: Record<BuiltinToolId, () => CapabilityDescriptor> = {
  exec: execCapability,
  hard_memory_query: hardMemoryCapability,
  hard_memory_write: hardMemoryWriteCapability,
  codex: codexCapability
};

export function buildCatalog(options: CatalogOptions = {}): CapabilityCatalog {
  const enabledTools = new Set<BuiltinToolId>(
    options.enabledTools ??
      (Object.keys(BUILTIN_CAPABILITIES) as BuiltinToolId[]).filter((toolId) => {
        if (toolId === "exec") {
          return options.enableExec !== false;
        }
        if (toolId === "hard_memory_query") {
          return options.enableHardMemoryQuery !== false;
        }
        if (toolId === "hard_memory_write") {
          return options.enableHardMemoryWrite !== false;
        }
        if (toolId === "codex") {
          return options.enableCodex !== false;
        }
        return true;
      })
  );
  const capabilities: CapabilityDescriptor[] = [];
  for (const toolId of Object.keys(BUILTIN_CAPABILITIES) as BuiltinToolId[]) {
    if (!enabledTools.has(toolId)) {
      continue;
    }
    capabilities.push(BUILTIN_CAPABILITIES[toolId]());
  }
  return assertCapabilityCatalog({
    catalog_version: 1,
    capabilities
  });
}

export function buildRuntimeCatalog(options: RuntimeCatalogOptions = {}): CapabilityCatalog {
  const capabilities: CapabilityDescriptor[] = [radarrCapability(), sonarrCapability()];
  const tavilyApiKey = options.secrets?.tools?.tavily?.api_key?.trim();
  if (tavilyApiKey) {
    capabilities.push(tavilyWebSearchCapability());
  }
  return assertCapabilityCatalog({
    catalog_version: 1,
    capabilities
  });
}
