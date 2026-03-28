import { FIXED_CHAPTARR_ROOT_FOLDER_PATH } from "./chaptarr-client.js";
import type { CapabilityCatalog, CapabilityDescriptor } from "./contracts.js";
import { assertCapabilityCatalog } from "./contracts.js";
import type { OpenClawToolSecrets } from "./config.js";
import {
  FIXED_RADARR_QUALITY_PROFILE_NAME,
  FIXED_RADARR_ROOT_FOLDER_PATH,
  FIXED_SONARR_QUALITY_PROFILE_NAME,
  FIXED_SONARR_ROOT_FOLDER_PATH,
} from "./servarr.js";

export type BuiltinToolId = "hard_memory_query" | "hard_memory_write";

export type CatalogOptions = {
  enabledTools?: BuiltinToolId[];
  enableHardMemoryQuery?: boolean;
  enableHardMemoryWrite?: boolean;
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

function executionSafetyClassForSideEffectClass(sideEffectClass: string | undefined): string {
  switch (sideEffectClass) {
    case "memory_read":
    case "network_read":
    case "service_read":
      return "read_only";
    default:
      return "approval_required";
  }
}

function hardMemoryCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.hard_memory_query",
    tool_surface_id: "vicuna.memory.hard_query",
    capability_kind: "memory_adapter",
    owner_plugin_id: "vicuna-memory",
    tool_name: "hard_memory_query",
    tool_family_id: "hard_memory",
    tool_family_name: "Hard Memory",
    tool_family_description: "Read from or write durable memory primitives in Vicuña hard memory.",
    method_name: "query",
    method_description: "Query Vicuña hard memory with a retrieval string.",
    description: "Query Vicuña hard memory with typed results",
    input_schema_json: {
      type: "object",
      required: ["query"],
      properties: {
        query: {
          type: "string",
          description:
            "The retrieval query to run against Vicuña hard memory when looking for relevant durable memories or prior tool observations."
        },
        limit: {
          type: "integer",
          description: "Optional result cap."
        },
        domain: {
          type: "string",
          description: "Optional domain filter."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_read",
    execution_safety_class: "read_only",
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
    dispatch_backend: "markdown_hard_memory"
  };
}

function hardMemoryWriteCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.hard_memory_write",
    tool_surface_id: "vicuna.memory.hard_write",
    capability_kind: "memory_adapter",
    owner_plugin_id: "vicuna-memory",
    tool_name: "hard_memory_write",
    tool_family_id: "hard_memory",
    tool_family_name: "Hard Memory",
    tool_family_description: "Read from or write durable memory primitives in Vicuña hard memory.",
    method_name: "write",
    method_description: "Archive a batch of durable memory primitives.",
    description: "Archive explicit durable memories to Vicuña markdown hard memory",
    input_schema_json: {
      type: "object",
      required: ["memories"],
      properties: {
        memories: {
          type: "array",
          description:
            "The batch of durable memory primitives to archive into Vicuña markdown hard memory.",
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
    execution_safety_class: "approval_required",
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
    dispatch_backend: "markdown_hard_memory"
  };
}

function skillReadCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.skill_read",
    tool_surface_id: "vicuna.skills.read",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-skills",
    tool_name: "skill_read",
    tool_family_id: "skills",
    tool_family_name: "Skills",
    tool_family_description: "Read or create host-owned skill markdown files through explicit runtime tools.",
    method_name: "read",
    method_description: "Read one skill markdown file by name.",
    description: "Read one host-owned skill markdown file by name so the runtime can load detailed instructions on demand instead of auto-injecting them.",
    input_schema_json: {
      type: "object",
      required: ["name"],
      properties: {
        name: {
          type: "string",
          description: "The skill file name or file stem to read."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "filesystem_read",
    execution_safety_class: "read_only",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-runtime/tool/skill_read",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_SIMULATION_SAFE,
      COG_TOOL_FLAG_REMEDIATION_SAFE
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "skills"
  };
}

function skillCreateCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.skill_create",
    tool_surface_id: "vicuna.skills.create",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-skills",
    tool_name: "skill_create",
    tool_family_id: "skills",
    tool_family_name: "Skills",
    tool_family_description: "Read or create host-owned skill markdown files through explicit runtime tools.",
    method_name: "create",
    method_description: "Create or update one skill markdown file by name.",
    description: "Create or update one host-owned skill markdown file. Only use this tool when the user directly asks to create or update a skill.",
    input_schema_json: {
      type: "object",
      required: ["name", "content"],
      properties: {
        name: {
          type: "string",
          description: "The requested skill name to write as a markdown file."
        },
        content: {
          type: "string",
          description: "The full markdown body to store in the skill file."
        },
        overwrite: {
          type: "boolean",
          description: "Whether an existing skill file may be updated in place."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "filesystem_write",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-runtime/tool/skill_create",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "skills"
  };
}

function tavilyWebSearchCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.tavily.web_search",
    tool_surface_id: "vicuna.web.search.tavily",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-tavily",
    tool_name: "web_search",
    tool_family_id: "web_search",
    tool_family_name: "Web Search",
    tool_family_description: "Search the live web and return source-grounded evidence.",
    method_name: "search",
    method_description: "Run one Tavily web search query.",
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
    execution_safety_class: "read_only",
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
    tool_family_id: "radarr",
    tool_family_name: "Radarr",
    tool_family_description:
      "Inspect and manage the Radarr movie library on the media server.",
    method_name: "generic",
    method_description: "Call the generic Radarr wrapper with an explicit action argument.",
    description:
      `Inspect and manage the Radarr movie library on the LAN-connected media server. Use this for Radarr system status, queue state, calendar, root folders, quality profiles, existing movies, movie lookup, and adding a movie into the fixed destination root ${FIXED_RADARR_ROOT_FOLDER_PATH} with the deployment-fixed quality profile ${FIXED_RADARR_QUALITY_PROFILE_NAME}.`,
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
            "inspect",
            "list_movies",
            "list_downloaded_movies",
            "lookup_movie",
            "add_movie",
            "download_movie",
            "delete_movie",
            "delete_movies",
          ],
          description:
            "The Radarr operation to perform. Read actions inspect or look up movies without starting downloads; add_movie is the legacy add path; download_movie starts acquisition by adding the movie if needed and then triggering Radarr's search flow; delete_movie removes one tracked movie and deletes its files by default."
        },
        term: {
          type: "string",
          description:
            "Search term for lookup_movie, add_movie, download_movie, or delete_movie. Use a precise movie title and optionally a year."
        },
        movie_id: {
          type: "integer",
          description:
            "Optional tracked Radarr movie id to delete directly without title resolution."
        },
        movie_ids: {
          type: "array",
          description:
            "Optional tracked Radarr movie ids to delete in one batch request.",
          items: {
            type: "integer",
            description: "One tracked Radarr movie id."
          }
        },
        tmdb_id: {
          type: "integer",
          description:
            "Optional TMDb movie id used to disambiguate the chosen lookup result when add_movie finds multiple candidates or to select the tracked movie to delete."
        },
        terms: {
          type: "array",
          description:
            "Optional precise movie title terms to delete in one batch request.",
          items: {
            type: "string",
            description: "One movie title term."
          }
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
        monitored: {
          type: "boolean",
          description:
            "Whether the added movie should be monitored in Radarr."
        },
        minimum_availability: {
          type: "string",
          enum: ["tba", "announced", "inCinemas", "released", "deleted"],
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
        },
        delete_files: {
          type: "boolean",
          description:
            "Whether delete_movie should remove the movie files from disk. Defaults to true."
        },
        add_import_exclusion: {
          type: "boolean",
          description:
            "Whether delete_movie should add an import exclusion for the removed movie."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "service_api",
    execution_safety_class: "approval_required",
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
    tool_family_id: "sonarr",
    tool_family_name: "Sonarr",
    tool_family_description:
      "Inspect and manage the Sonarr television library on the media server.",
    method_name: "generic",
    method_description: "Call the generic Sonarr wrapper with an explicit action argument.",
    description:
      `Inspect and manage the Sonarr series library on the LAN-connected media server. Use this for Sonarr system status, queue state, calendar, root folders, quality profiles, existing series, series lookup, and adding a series into the fixed destination root ${FIXED_SONARR_ROOT_FOLDER_PATH} with the deployment-fixed quality profile ${FIXED_SONARR_QUALITY_PROFILE_NAME}.`,
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
            "inspect",
            "list_series",
            "list_downloaded_series",
            "lookup_series",
            "add_series",
            "download_series",
            "delete_series",
          ],
          description:
            "The Sonarr operation to perform. Read actions inspect or look up series without starting downloads; add_series is the legacy add path; download_series starts acquisition by adding the series if needed and then triggering Sonarr's search flow; delete_series removes one tracked series and deletes its files by default."
        },
        term: {
          type: "string",
          description:
            "Search term for lookup_series, add_series, download_series, or delete_series. Use a precise series title and optionally a year."
        },
        series_id: {
          type: "integer",
          description:
            "Optional tracked Sonarr series id to delete directly without title resolution."
        },
        series_ids: {
          type: "array",
          description:
            "Optional tracked Sonarr series ids to delete in one batch request.",
          items: {
            type: "integer",
            description: "One tracked Sonarr series id."
          }
        },
        tvdb_id: {
          type: "integer",
          description:
            "Optional TVDb series id used to disambiguate the chosen lookup result when add_series finds multiple candidates or to select the tracked series to delete."
        },
        tmdb_id: {
          type: "integer",
          description:
            "Optional TMDb series id used to disambiguate the chosen lookup result when add_series finds multiple candidates or to select the tracked series to delete."
        },
        terms: {
          type: "array",
          description:
            "Optional precise series title terms to delete in one batch request.",
          items: {
            type: "string",
            description: "One series title term."
          }
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
        },
        delete_files: {
          type: "boolean",
          description:
            "Whether delete_series should remove the series files from disk. Defaults to true."
        },
        add_import_list_exclusion: {
          type: "boolean",
          description:
            "Whether delete_series should add an import-list exclusion for the removed series."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "service_api",
    execution_safety_class: "approval_required",
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

function chaptarrCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.servarr.chaptarr",
    tool_surface_id: "vicuna.media.chaptarr",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-servarr",
    tool_name: "chaptarr",
    tool_family_id: "chaptarr",
    tool_family_name: "Chaptarr",
    tool_family_description:
      "Inspect and manage the Chaptarr ebook library on the media server.",
    method_name: "generic",
    method_description: "Call the generic Chaptarr wrapper with an explicit action argument.",
    description:
      "Inspect and manage the Chaptarr ebook library on the LAN-connected media server. Use this for Chaptarr system and health status, queue status, root folders, ebook profile lookup, Hardcover-backed mixed search, author or book lookup, scoped book or series inspection, and starting ebook acquisition flows.",
    input_schema_json: {
      type: "object",
      required: ["action"],
      properties: {
        action: {
          type: "string",
          enum: [
            "system_status",
            "health",
            "queue_status",
            "root_folders",
            "quality_profiles",
            "metadata_profiles",
            "inspect",
            "list_authors",
            "search",
            "author_lookup",
            "book_lookup",
            "list_books",
            "list_downloaded_books",
            "list_series",
            "add_author",
            "add_book",
            "download_author",
            "download_book",
            "delete_book",
            "delete_books",
          ],
          description:
            `The Chaptarr operation to perform. Read actions inspect or look up authors and books without starting downloads; inspect is a generic alias for listing current authors; search and book_lookup only surface ebook-capable book options while preserving author discovery; add_author, add_book, download_author, and download_book all start ebook acquisition using the fixed NAS Books root at ${FIXED_CHAPTARR_ROOT_FOLDER_PATH}; download_book resolves the ebook candidate and starts acquisition in the same tool call; delete_book removes one tracked ebook and deletes its files by default.`
        },
        term: {
          type: "string",
          description:
            "Search term for search, author_lookup, book_lookup, add_author, add_book, download_book, or delete_book. Book-searching actions only surface ebook-capable book options for the supplied phrase."
        },
        provider: {
          type: "string",
          description:
            "Optional upstream search provider for the generic search action. Omit it to let Chaptarr use its configured default, which is Hardcover on the current deployment."
        },
        foreign_author_id: {
          type: "string",
          description:
            "Optional foreign author id used to disambiguate the selected lookup result when add_author finds multiple candidates."
        },
        foreign_book_id: {
          type: "string",
          description:
            "Optional foreign book id used to disambiguate the selected lookup result when add_book finds multiple candidates or to select the tracked book to delete."
        },
        book_id: {
          type: "integer",
          description:
            "Optional tracked Chaptarr local book id to delete directly without title resolution."
        },
        book_ids: {
          type: "array",
          description:
            "Optional tracked Chaptarr local book ids to delete in one batch request.",
          items: {
            type: "integer",
            description: "One tracked Chaptarr local book id."
          }
        },
        foreign_edition_id: {
          type: "string",
          description:
            "Optional foreign edition id used to disambiguate the selected edition when add_book finds multiple candidates."
        },
        author_id: {
          type: "integer",
          description:
            "Optional Chaptarr author id used to scope list_books or list_series to a single author."
        },
        terms: {
          type: "array",
          description:
            "Optional precise book title terms to delete in one batch request.",
          items: {
            type: "string",
            description: "One book title term."
          }
        },
        media_type: {
          type: "string",
          enum: ["ebook"],
          description:
            "Media type selector for Chaptarr profile inspection. This deployment only exposes ebook handling."
        },
        quality_profile_id: {
          type: "integer",
          description:
            "Chaptarr ebook quality profile id to assign when adding an author or book. Required for add_author and add_book."
        },
        metadata_profile_id: {
          type: "integer",
          description:
            "Chaptarr metadata profile id to assign when adding an author or book. Required for add_author and add_book."
        },
        monitored: {
          type: "boolean",
          description:
            "Whether the added author or added book should be monitored in Chaptarr."
        },
        any_edition_ok: {
          type: "boolean",
          description:
            "Whether add_book may accept any edition instead of requiring one specific monitored edition."
        },
        monitor_new_items: {
          type: "string",
          enum: ["all", "none", "new"],
          description:
            "How Chaptarr should monitor newly discovered items for the added author."
        },
        monitor: {
          type: "string",
          enum: ["all", "future", "missing", "existing", "latest", "first", "none", "unknown"],
          description:
            "Chaptarr author-monitor mode used inside add_author addOptions."
        },
        search_for_missing_books: {
          type: "boolean",
          description:
            "Whether Chaptarr should immediately search for missing books after adding the author."
        },
        search_for_new_book: {
          type: "boolean",
          description:
            "Whether add_book should immediately search indexers and download clients for the newly added book."
        },
        tags: {
          type: "array",
          description:
            "Optional Chaptarr tag ids to assign to the author when adding it.",
          items: {
            type: "integer",
            description: "One Chaptarr tag id."
          }
        },
        delete_files: {
          type: "boolean",
          description:
            "Whether delete_book should remove the ebook files from disk. Defaults to true."
        },
        add_import_list_exclusion: {
          type: "boolean",
          description:
            "Whether delete_book should add an import-list exclusion for the removed tracked book."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "service_api",
    execution_safety_class: "approval_required",
    approval_mode: "policy_driven",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/openclaw-servarr/tool/chaptarr",
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

function ongoingTasksCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.ongoing_tasks",
    tool_surface_id: "vicuna.tasks.ongoing",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-ongoing-tasks",
    tool_name: "ongoing_tasks",
    tool_family_id: "ongoing_tasks",
    tool_family_name: "Ongoing Tasks",
    tool_family_description:
      "Create, inspect, update, complete, and delete recurring tasks that the system should poll again later.",
    method_name: "generic",
    method_description: "Run one ongoing-task registry action.",
    description:
      "Manage the hard-memory-backed recurring task registry that the control loop can poll for due work.",
    input_schema_json: {
      type: "object",
      required: ["action"],
      properties: {
        action: {
          type: "string",
          enum: ["create", "get", "edit", "delete", "complete"],
          description:
            "The ongoing-task action to perform. Create adds a new recurring task, get reads one or more tasks, edit changes the task text or cadence, delete removes the task from the active registry, and complete records that the task was just run."
        },
        task_id: {
          type: "string",
          description: "Stable ongoing-task id used by get, edit, delete, or complete."
        },
        task_text: {
          type: "string",
          description: "The actual recurring work instruction the system should perform."
        },
        interval: {
          type: "integer",
          minimum: 1,
          description: "The positive recurrence interval for the task cadence."
        },
        unit: {
          type: "string",
          enum: ["hours", "days", "weeks"],
          description: "The recurrence unit paired with interval."
        },
        due_only: {
          type: "boolean",
          description: "Whether get should return only currently due tasks."
        },
        active: {
          type: "boolean",
          description: "Whether the task should remain active for future due polls."
        },
        completed_at: {
          type: "string",
          description: "Optional ISO-8601 completion timestamp to record instead of now."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_write",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-ongoing-tasks/tool/ongoing_tasks",
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

function telegramRelayCapability(): CapabilityDescriptor {
  const telegramRelayWriteup =
    "Simple text is OK. For richer Telegram-native output, send request={method,payload}. " +
    "Prefer sendMessage with parse_mode plus reply_markup for most rich text and button replies. " +
    "Use reply_markup.inline_keyboard for inline buttons and other Telegram reply markup only when needed. " +
    "Prefer parse_mode over raw entity offsets unless you intentionally provide valid UTF-16-based entities.";
  return {
    capability_id: "openclaw.vicuna.telegram_relay",
    tool_surface_id: "vicuna.telegram.relay",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-telegram",
    tool_name: "telegram_relay",
    tool_family_id: "telegram",
    tool_family_name: "Telegram",
    tool_family_description:
      "Send direct user-facing follow-up messages through the provider-only Telegram bridge outbox.",
    method_name: "relay",
    method_description: "Queue one Telegram follow-up message as plain text or a structured Bot API send request.",
    description:
      `Send one direct follow-up message to a Telegram user through the provider-only bridge outbox without reviving the older ask-with-options or approval runtime. ${telegramRelayWriteup}`,
    input_schema_json: {
      type: "object",
      anyOf: [
        { required: ["text"] },
        { required: ["request"] }
      ],
      properties: {
        text: {
          type: "string",
          description: "Optional simple plain-text Telegram message. Use this when rich formatting or custom Telegram UI is unnecessary."
        },
        request: {
          type: "object",
          description:
            `Optional structured Telegram Bot API send request. ${telegramRelayWriteup} Do not include chat_id; the relay fills routing.`,
          required: ["method", "payload"],
          properties: {
            method: {
              type: "string",
              description:
                "Allowed outbound Telegram method, such as sendMessage, sendPhoto, sendDocument, sendVideo, sendAnimation, sendAudio, sendVoice, sendSticker, sendMediaGroup, sendLocation, sendVenue, sendContact, sendPoll, or sendDice."
            },
            payload: {
              type: "object",
              description:
                "Telegram Bot API payload for the selected method. Prefer sendMessage plus parse_mode, link_preview_options, and reply_markup for most rich text/button replies."
            }
          }
        },
        chat_scope: {
          type: "string",
          description:
            "The Telegram chat scope to route the follow-up into. Optional only when the wrapper has a configured default chat scope."
        },
        reply_to_message_id: {
          type: "integer",
          minimum: 1,
          description: "Optional Telegram message id to use as the reply anchor."
        },
        intent: {
          type: "string",
          description: "Optional intent label that classifies the follow-up, such as question or conclusion."
        },
        dedupe_key: {
          type: "string",
          description: "Optional dedupe key used to suppress duplicate queued follow-ups for the same chat."
        },
        urgency: {
          type: "number",
          minimum: 0,
          description: "Optional normalized urgency score for the queued follow-up."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "user_contact",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-telegram/tool/telegram_relay",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_bash"
  };
}

function parsedDocumentsSearchCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.parsed-documents.search-chunks",
    tool_surface_id: "vicuna.documents.parsed.search_chunks",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-parsed-documents",
    tool_name: "parsed_documents_search_chunks",
    tool_family_id: "parsed_documents",
    tool_family_name: "Parsed Documents",
    tool_family_description:
      "Search locally stored parsed-document chunks derived from Telegram-uploaded files and return compact labeled matches.",
    method_name: "search_chunks",
    method_description: "Search locally stored parsed-document chunks by explicit lexical query.",
    description:
      "Search the local parsed-document chunk store under the Vicuña docs directory and return only the compact matching chunks the system needs, labeled with their source document titles.",
    input_schema_json: {
      type: "object",
      required: ["query"],
      properties: {
        query: {
          type: "string",
          description: "The semantic retrieval query to run against stored parsed-document chunks."
        },
        limit: {
          type: "integer",
          minimum: 1,
          maximum: 8,
          description: "Optional maximum number of matching chunks to return."
        },
        threshold: {
          type: "number",
          minimum: 0,
          maximum: 1,
          description:
            "Optional similarity threshold override in the range [0, 1]. When omitted, the wrapper applies stricter defaults for shorter queries."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_read",
    execution_safety_class: "read_only",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-parsed-documents/tool/parsed_documents_search_chunks",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_SIMULATION_SAFE,
      COG_TOOL_FLAG_REMEDIATION_SAFE
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_bash"
  };
}

type CapabilitySchema = {
  type: "object";
  required?: string[];
  properties: Record<string, unknown>;
};

type CapabilityAliasOptions = {
  capabilityId: string;
  toolSurfaceId: string;
  toolName: string;
  toolFamilyId?: string;
  toolFamilyName?: string;
  toolFamilyDescription?: string;
  methodName: string;
  methodDescription: string;
  description: string;
  fixedArguments: Record<string, unknown>;
  includeProperties: string[];
  requiredProperties?: string[];
  provenanceSuffix?: string;
  sideEffectClass?: string;
  executionSafetyClass?: string;
  toolFlags?: number;
};

function pickSchemaProperties(schema: CapabilitySchema, keys: string[]): Record<string, unknown> {
  const picked: Record<string, unknown> = {};
  for (const key of keys) {
    if (key in schema.properties) {
      picked[key] = schema.properties[key];
    }
  }
  return picked;
}

function makeCapabilityAlias(
  base: CapabilityDescriptor,
  options: CapabilityAliasOptions
): CapabilityDescriptor {
  const schema = base.input_schema_json as CapabilitySchema;
  const sideEffectClass = options.sideEffectClass ?? base.side_effect_class;
  const requiredFromBase = new Set(
    Array.isArray(schema.required) ? schema.required.filter((entry) => entry !== "action") : []
  );
  const requiredProperties =
    options.requiredProperties ??
    options.includeProperties.filter((key) => requiredFromBase.has(key));
  return {
    ...base,
    capability_id: options.capabilityId,
    tool_surface_id: options.toolSurfaceId,
    tool_name: options.toolName,
    tool_family_id: options.toolFamilyId ?? base.tool_family_id ?? base.tool_name,
    tool_family_name: options.toolFamilyName ?? base.tool_family_name ?? base.tool_name,
    tool_family_description:
      options.toolFamilyDescription ?? base.tool_family_description ?? base.description,
    method_name: options.methodName,
    method_description: options.methodDescription,
    description: options.description,
    input_schema_json: {
      type: "object",
      ...(requiredProperties.length > 0 ? { required: requiredProperties } : {}),
      properties: pickSchemaProperties(schema, options.includeProperties)
    },
    fixed_arguments_json: options.fixedArguments,
    side_effect_class: sideEffectClass,
    execution_safety_class:
      options.executionSafetyClass ??
      executionSafetyClassForSideEffectClass(sideEffectClass),
    tool_flags: options.toolFlags ?? base.tool_flags,
    provenance_namespace: `${base.provenance_namespace}/${options.provenanceSuffix ?? options.toolName}`
  };
}

function mediaRuntimeCapabilities(): CapabilityDescriptor[] {
  const radarr = radarrCapability();
  const sonarr = sonarrCapability();
  const chaptarr = chaptarrCapability();
  const readOnlyToolFlags = combineToolFlags(
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
    COG_TOOL_FLAG_DMN_ELIGIBLE,
    COG_TOOL_FLAG_SIMULATION_SAFE,
    COG_TOOL_FLAG_REMEDIATION_SAFE
  );
  const acquisitionToolFlags = combineToolFlags(
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
    COG_TOOL_FLAG_DMN_ELIGIBLE,
    COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
  );

  return [
    makeCapabilityAlias(radarr, {
      capabilityId: "openclaw.servarr.radarr.list-downloaded-movies",
      toolSurfaceId: "vicuna.media.radarr.list_downloaded_movies",
      toolName: "radarr_list_downloaded_movies",
      methodName: "list_downloaded_movies",
      methodDescription: "List the movies already fully downloaded in Radarr.",
      description: "List only the movies that are already fully downloaded in Radarr. The returned payload is limited to downloaded-movie summaries.",
      fixedArguments: { action: "list_downloaded_movies" },
      includeProperties: [],
      sideEffectClass: "service_read",
      toolFlags: readOnlyToolFlags,
    }),
    makeCapabilityAlias(radarr, {
      capabilityId: "openclaw.servarr.radarr.download-movie",
      toolSurfaceId: "vicuna.media.radarr.download_movie",
      toolName: "radarr_download_movie",
      methodName: "download_movie",
      methodDescription: "Start Radarr movie acquisition for the requested title.",
      description: `Start Radarr movie acquisition for the requested title. If the movie is not yet tracked, this method adds it into the fixed destination root ${FIXED_RADARR_ROOT_FOLDER_PATH} with the deployment-fixed quality profile ${FIXED_RADARR_QUALITY_PROFILE_NAME} and requests an immediate search. If the movie is already tracked, this method triggers Radarr's native Movies Search command for the existing item.`,
      fixedArguments: { action: "download_movie" },
      includeProperties: [
        "term",
        "tmdb_id",
        "monitored",
        "minimum_availability",
        "monitor",
        "search_for_movie",
        "tags",
      ],
      requiredProperties: ["term"],
      sideEffectClass: "service_acquisition",
      toolFlags: acquisitionToolFlags,
    }),
    makeCapabilityAlias(radarr, {
      capabilityId: "openclaw.servarr.radarr.delete-movies",
      toolSurfaceId: "vicuna.media.radarr.delete_movies",
      toolName: "radarr_delete_movies",
      methodName: "delete_movies",
      methodDescription: "Delete one or more downloaded Radarr movies and remove their files from disk.",
      description: "Delete one or more tracked Radarr movies from the library and remove their files from `/movies` by default. Resolve targets by tracked ids, TMDb id, or precise title terms.",
      fixedArguments: { action: "delete_movies" },
      includeProperties: ["term", "terms", "movie_id", "movie_ids", "tmdb_id", "delete_files", "add_import_exclusion"],
      sideEffectClass: "service_api",
      toolFlags: acquisitionToolFlags,
    }),
    makeCapabilityAlias(sonarr, {
      capabilityId: "openclaw.servarr.sonarr.list-downloaded-series",
      toolSurfaceId: "vicuna.media.sonarr.list_downloaded_series",
      toolName: "sonarr_list_downloaded_series",
      methodName: "list_downloaded_series",
      methodDescription: "List the series already fully downloaded in Sonarr.",
      description: "List only the series that already have downloaded episode files in Sonarr. The returned payload is limited to downloaded-series summaries.",
      fixedArguments: { action: "list_downloaded_series" },
      includeProperties: [],
      sideEffectClass: "service_read",
      toolFlags: readOnlyToolFlags,
    }),
    makeCapabilityAlias(sonarr, {
      capabilityId: "openclaw.servarr.sonarr.download-series",
      toolSurfaceId: "vicuna.media.sonarr.download_series",
      toolName: "sonarr_download_series",
      methodName: "download_series",
      methodDescription: "Start Sonarr series acquisition for the requested title.",
      description: `Start Sonarr series acquisition for the requested title. If the series is not yet tracked, this method adds it into the fixed destination root ${FIXED_SONARR_ROOT_FOLDER_PATH} with the deployment-fixed quality profile ${FIXED_SONARR_QUALITY_PROFILE_NAME} and requests an immediate missing-episode search. If the series is already tracked, this method triggers Sonarr's native Series Search command for the existing item.`,
      fixedArguments: { action: "download_series" },
      includeProperties: [
        "term",
        "tvdb_id",
        "tmdb_id",
        "monitored",
        "season_folder",
        "series_type",
        "monitor_new_items",
        "monitor",
        "search_for_missing_episodes",
        "search_for_cutoff_unmet_episodes",
        "tags",
      ],
      requiredProperties: ["term"],
      sideEffectClass: "service_acquisition",
      toolFlags: acquisitionToolFlags,
    }),
    makeCapabilityAlias(sonarr, {
      capabilityId: "openclaw.servarr.sonarr.delete-series",
      toolSurfaceId: "vicuna.media.sonarr.delete_series",
      toolName: "sonarr_delete_series",
      methodName: "delete_series",
      methodDescription: "Delete one or more downloaded Sonarr series and remove their files from disk.",
      description: "Delete one or more tracked Sonarr series from the library and remove their files from `/tv` by default. Resolve targets by tracked ids, TVDb id, TMDb id, or precise title terms.",
      fixedArguments: { action: "delete_series" },
      includeProperties: ["term", "terms", "series_id", "series_ids", "tvdb_id", "tmdb_id", "delete_files", "add_import_list_exclusion"],
      sideEffectClass: "service_api",
      toolFlags: acquisitionToolFlags,
    }),
    makeCapabilityAlias(chaptarr, {
      capabilityId: "openclaw.servarr.chaptarr.list-downloaded-books",
      toolSurfaceId: "vicuna.media.chaptarr.list_downloaded_books",
      toolName: "chaptarr_list_downloaded_books",
      methodName: "list_downloaded_books",
      methodDescription: "List the books already fully downloaded in Chaptarr.",
      description: "List only the ebook books that are already fully downloaded in Chaptarr. The returned payload is limited to downloaded-book summaries.",
      fixedArguments: { action: "list_downloaded_books" },
      includeProperties: [],
      sideEffectClass: "service_read",
      toolFlags: readOnlyToolFlags,
    }),
    makeCapabilityAlias(chaptarr, {
      capabilityId: "openclaw.servarr.chaptarr.download-book",
      toolSurfaceId: "vicuna.media.chaptarr.download_book",
      toolName: "chaptarr_download_book",
      methodName: "download_book",
      methodDescription: "Start ebook-only Chaptarr acquisition for a specific title in one tool call.",
      description: `Start ebook-only Chaptarr acquisition for a specific title. If the book is not yet tracked, this method resolves an ebook-capable candidate, adds it using the fixed NAS Books folder at ${FIXED_CHAPTARR_ROOT_FOLDER_PATH}, and requests an immediate Chaptarr search. If the direct ebook path cannot be resolved, it falls back to creating or repairing a monitored missing/wanted book and immediately waking the legal importer when available. The result always reports the high-level acquisition or importer state instead of raw API payloads.`,
      fixedArguments: { action: "download_book" },
      includeProperties: ["term", "foreign_book_id", "foreign_edition_id"],
      requiredProperties: ["term"],
      sideEffectClass: "service_acquisition",
      toolFlags: acquisitionToolFlags,
    }),
    makeCapabilityAlias(chaptarr, {
      capabilityId: "openclaw.servarr.chaptarr.delete-books",
      toolSurfaceId: "vicuna.media.chaptarr.delete_books",
      toolName: "chaptarr_delete_books",
      methodName: "delete_books",
      methodDescription: "Delete one or more downloaded Chaptarr books and remove their files from disk.",
      description: "Delete one or more tracked Chaptarr ebook books from the library and remove their files from `/books` by default. Resolve targets by tracked ids, foreign book id, or precise title terms.",
      fixedArguments: { action: "delete_books" },
      includeProperties: ["term", "terms", "book_id", "book_ids", "foreign_book_id", "delete_files", "add_import_list_exclusion"],
      sideEffectClass: "service_api",
      toolFlags: acquisitionToolFlags,
    }),
  ];
}

function ongoingTaskRuntimeCapabilities(): CapabilityDescriptor[] {
  const ongoingTasks = ongoingTasksCapability();
  const readOnlyToolFlags = combineToolFlags(
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
    COG_TOOL_FLAG_DMN_ELIGIBLE,
    COG_TOOL_FLAG_SIMULATION_SAFE,
    COG_TOOL_FLAG_REMEDIATION_SAFE
  );
  const mutationToolFlags = combineToolFlags(
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
    COG_TOOL_FLAG_DMN_ELIGIBLE,
    COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
  );

  return [
    makeCapabilityAlias(ongoingTasks, {
      capabilityId: "openclaw.vicuna.ongoing-tasks.create",
      toolSurfaceId: "vicuna.tasks.ongoing.create",
      toolName: "ongoing_tasks_create",
      methodName: "create",
      methodDescription: "Create one recurring task in the hard-memory-backed registry.",
      description:
        "Create one recurring task with explicit task text and cadence fields. The returned payload is limited to the compact task summary the system needs for future polling.",
      fixedArguments: { action: "create" },
      includeProperties: ["task_text", "interval", "unit"],
      requiredProperties: ["task_text", "interval", "unit"],
      sideEffectClass: "memory_write",
      toolFlags: mutationToolFlags,
    }),
    makeCapabilityAlias(ongoingTasks, {
      capabilityId: "openclaw.vicuna.ongoing-tasks.get",
      toolSurfaceId: "vicuna.tasks.ongoing.get",
      toolName: "ongoing_tasks_get",
      methodName: "get",
      methodDescription: "Fetch one ongoing task or list active recurring tasks.",
      description:
        "Fetch one ongoing task by id or list the active recurring tasks with compact summaries only.",
      fixedArguments: { action: "get" },
      includeProperties: ["task_id"],
      sideEffectClass: "memory_read",
      toolFlags: readOnlyToolFlags,
    }),
    makeCapabilityAlias(ongoingTasks, {
      capabilityId: "openclaw.vicuna.ongoing-tasks.get-due",
      toolSurfaceId: "vicuna.tasks.ongoing.get_due",
      toolName: "ongoing_tasks_get_due",
      methodName: "get_due",
      methodDescription: "List only the recurring tasks whose due time has already passed.",
      description:
        "Return only the ongoing tasks that are currently due according to explicit interval math in the wrapper.",
      fixedArguments: { action: "get", due_only: true },
      includeProperties: [],
      sideEffectClass: "memory_read",
      toolFlags: readOnlyToolFlags,
    }),
    makeCapabilityAlias(ongoingTasks, {
      capabilityId: "openclaw.vicuna.ongoing-tasks.edit",
      toolSurfaceId: "vicuna.tasks.ongoing.edit",
      toolName: "ongoing_tasks_edit",
      methodName: "edit",
      methodDescription: "Update a recurring task's text, cadence, or active state.",
      description:
        "Update one recurring task and return its compact summary with refreshed due-state fields.",
      fixedArguments: { action: "edit" },
      includeProperties: ["task_id", "task_text", "interval", "unit", "active"],
      requiredProperties: ["task_id"],
      sideEffectClass: "memory_write",
      toolFlags: mutationToolFlags,
    }),
    makeCapabilityAlias(ongoingTasks, {
      capabilityId: "openclaw.vicuna.ongoing-tasks.delete",
      toolSurfaceId: "vicuna.tasks.ongoing.delete",
      toolName: "ongoing_tasks_delete",
      methodName: "delete",
      methodDescription: "Delete one recurring task from the active registry.",
      description:
        "Delete one recurring task by id and return only the deletion status the system needs.",
      fixedArguments: { action: "delete" },
      includeProperties: ["task_id"],
      requiredProperties: ["task_id"],
      sideEffectClass: "memory_write",
      toolFlags: mutationToolFlags,
    }),
    makeCapabilityAlias(ongoingTasks, {
      capabilityId: "openclaw.vicuna.ongoing-tasks.complete",
      toolSurfaceId: "vicuna.tasks.ongoing.complete",
      toolName: "ongoing_tasks_complete",
      methodName: "complete",
      methodDescription: "Record that one recurring task was just completed.",
      description:
        "Advance one recurring task's last-done timestamp after the system finishes the work.",
      fixedArguments: { action: "complete" },
      includeProperties: ["task_id", "completed_at"],
      requiredProperties: ["task_id"],
      sideEffectClass: "memory_write",
      toolFlags: mutationToolFlags,
    }),
  ];
}

const BUILTIN_CAPABILITIES: Record<BuiltinToolId, () => CapabilityDescriptor> = {
  hard_memory_query: hardMemoryCapability,
  hard_memory_write: hardMemoryWriteCapability,
};

export function buildCatalog(options: CatalogOptions = {}): CapabilityCatalog {
  const enabledTools = new Set<BuiltinToolId>(
    options.enabledTools ??
      (Object.keys(BUILTIN_CAPABILITIES) as BuiltinToolId[]).filter((toolId) => {
        if (toolId === "hard_memory_query") {
          return options.enableHardMemoryQuery !== false;
        }
        if (toolId === "hard_memory_write") {
          return options.enableHardMemoryWrite !== false;
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
  const capabilities: CapabilityDescriptor[] = [
    {
      capability_id: "openclaw.vicuna.media.read",
      tool_surface_id: "vicuna.media.read",
      capability_kind: "tool",
      owner_plugin_id: "vicuna-runtime",
      tool_name: "media_read",
      tool_family_id: "media",
      tool_family_name: "Media",
      tool_family_description: "Read from the media server or parsed-document store through one direct tool.",
      method_name: "read",
      method_description: "Read movie, series, book, or document state through one direct tool call.",
      description: "Read media-server or parsed-document state through one direct tool call. Use movie, series, or book for library/media lookup and document for parsed-document retrieval.",
      input_schema_json: {
        type: "object",
        required: ["media_kind", "query"],
        properties: {
          media_kind: {
            type: "string",
            enum: ["movie", "series", "book", "document"],
            description: "Which media domain to inspect."
          },
          query: {
            type: "string",
            description: "Natural-language lookup text."
          },
          backend_hint: {
            type: "string",
            enum: ["auto", "radarr", "sonarr", "chaptarr", "parsed_documents"],
            description: "Optional backend preference. Use auto unless you have a concrete reason to override."
          },
          limit: {
            type: "integer",
            description: "Optional result cap for read responses."
          },
          status_filter: {
            type: "string",
            enum: ["all", "downloaded", "missing", "upcoming"],
            description: "Optional status filter that narrows the read behavior."
          }
        }
      },
      output_contract: "completed_result",
      side_effect_class: "service_read",
      execution_safety_class: "read_only",
      approval_mode: "none",
      execution_modes: ["sync"],
      provenance_namespace: "openclaw/vicuna-runtime/tool/media_read",
      tool_kind: 4,
      tool_flags: combineToolFlags(
        COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
        COG_TOOL_FLAG_DMN_ELIGIBLE,
        COG_TOOL_FLAG_SIMULATION_SAFE,
        COG_TOOL_FLAG_REMEDIATION_SAFE
      ),
      latency_class: 1,
      max_steps_reserved: 2,
      dispatch_backend: "flattened_media_read"
    },
    {
      capability_id: "openclaw.vicuna.media.download",
      tool_surface_id: "vicuna.media.download",
      capability_kind: "tool",
      owner_plugin_id: "vicuna-runtime",
      tool_name: "media_download",
      tool_family_id: "media",
      tool_family_name: "Media",
      tool_family_description: "Start media acquisition through one direct tool.",
      method_name: "download",
      method_description: "Start movie, series, or book acquisition through one direct tool call.",
      description: "Start movie, series, or ebook acquisition through one direct tool call.",
      input_schema_json: {
        type: "object",
        required: ["media_kind", "query"],
        properties: {
          media_kind: {
            type: "string",
            enum: ["movie", "series", "book"],
            description: "Which media domain to modify."
          },
          query: {
            type: "string",
            description: "Natural-language title or search string."
          },
          backend_hint: {
            type: "string",
            enum: ["auto", "radarr", "sonarr", "chaptarr"],
            description: "Optional backend preference. Use auto unless you need an explicit backend."
          },
          id_hint: {
            type: "string",
            description: "Optional external identifier used to disambiguate the selected item."
          }
        }
      },
      output_contract: "completed_result",
      side_effect_class: "service_acquisition",
      execution_safety_class: "approval_required",
      approval_mode: "policy_driven",
      execution_modes: ["sync"],
      provenance_namespace: "openclaw/vicuna-runtime/tool/media_download",
      tool_kind: 4,
      tool_flags: combineToolFlags(
        COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
        COG_TOOL_FLAG_DMN_ELIGIBLE,
        COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
      ),
      latency_class: 1,
      max_steps_reserved: 3,
      dispatch_backend: "flattened_media_download"
    },
    {
      capability_id: "openclaw.vicuna.media.delete",
      tool_surface_id: "vicuna.media.delete",
      capability_kind: "tool",
      owner_plugin_id: "vicuna-runtime",
      tool_name: "media_delete",
      tool_family_id: "media",
      tool_family_name: "Media",
      tool_family_description: "Delete tracked media through one direct tool.",
      method_name: "delete",
      method_description: "Delete a movie, series, or book through one direct tool call.",
      description: "Delete tracked media through one direct tool call while keeping runtime-side validation explicit.",
      input_schema_json: {
        type: "object",
        required: ["media_kind", "query"],
        properties: {
          media_kind: {
            type: "string",
            enum: ["movie", "series", "book"],
            description: "Which media domain to modify."
          },
          query: {
            type: "string",
            description: "Natural-language title or identifier hint."
          },
          backend_hint: {
            type: "string",
            enum: ["auto", "radarr", "sonarr", "chaptarr"],
            description: "Optional backend preference. Use auto unless you need an explicit backend."
          },
          delete_files: {
            type: "boolean",
            description: "Whether files should also be deleted."
          }
        }
      },
      output_contract: "completed_result",
      side_effect_class: "service_api",
      execution_safety_class: "approval_required",
      approval_mode: "policy_driven",
      execution_modes: ["sync"],
      provenance_namespace: "openclaw/vicuna-runtime/tool/media_delete",
      tool_kind: 4,
      tool_flags: combineToolFlags(
        COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
        COG_TOOL_FLAG_DMN_ELIGIBLE,
        COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
      ),
      latency_class: 1,
      max_steps_reserved: 3,
      dispatch_backend: "flattened_media_delete"
    },
    {
      ...hardMemoryCapability(),
      capability_id: "openclaw.vicuna.hard_memory_read",
      tool_surface_id: "vicuna.memory.hard_read",
      tool_name: "hard_memory_read",
      method_name: "read",
      method_description: "Read Vicuña hard memory with a retrieval query.",
      description: "Read durable memory primitives from Vicuña hard memory.",
      provenance_namespace: "openclaw/vicuna-memory/memory_adapter/hard_memory_read"
    },
    {
      ...hardMemoryWriteCapability(),
      capability_id: "openclaw.vicuna.hard_memory_write_flattened",
      tool_surface_id: "vicuna.memory.hard_write",
      tool_name: "hard_memory_write",
      method_name: "write",
      method_description: "Archive one batch of durable memory primitives.",
      description: "Archive explicit durable memories to Vicuña markdown hard memory through one direct tool call.",
      provenance_namespace: "openclaw/vicuna-memory/memory_adapter/hard_memory_write_flattened"
    },
    skillReadCapability(),
    skillCreateCapability(),
    {
      capability_id: "openclaw.vicuna.host_shell",
      tool_surface_id: "vicuna.host.shell",
      capability_kind: "tool",
      owner_plugin_id: "vicuna-runtime",
      tool_name: "host_shell",
      tool_family_id: "host_shell",
      tool_family_name: "Host Shell",
      tool_family_description: "A last-resort host shell fallback for direct file or shell manipulation in the runtime's own workspace when specialized tools do not fit.",
      method_name: "execute",
      method_description: "Run one bounded shell command in the runtime's host workspace and return a structured observation.",
      description: "Last-resort host shell fallback for direct file or shell manipulation in the runtime's own host workspace. Prefer media_read, media_download, media_delete, hard_memory_read, hard_memory_write, skill_read, skill_create, ongoing_task_create, ongoing_task_delete, and web_search whenever they fit the task. Use host_shell only when you need direct host-side file or shell operations that those specialized tools do not provide.",
      input_schema_json: {
        type: "object",
        description: "Payload for one bounded host shell invocation rooted in the runtime's dedicated host workspace.",
        required: ["command", "purpose"],
        properties: {
          command: {
            type: "string",
            description: "The shell command to execute."
          },
          purpose: {
            type: "string",
            description: "A short description of why this host-shell command is needed."
          },
          working_directory: {
            type: "string",
            description: "Optional relative subdirectory under the host shell workspace to use as the starting directory."
          },
          timeout_ms: {
            type: "integer",
            description: "Optional timeout in milliseconds for the shell command."
          }
        }
      },
      output_contract: "completed_result",
      side_effect_class: "filesystem_write",
      execution_safety_class: "approval_required",
      approval_mode: "policy_driven",
      execution_modes: ["sync"],
      provenance_namespace: "openclaw/vicuna-runtime/tool/host_shell",
      tool_kind: 4,
      tool_flags: combineToolFlags(
        COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
        COG_TOOL_FLAG_DMN_ELIGIBLE,
        COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
      ),
      latency_class: 1,
      max_steps_reserved: 3,
      dispatch_backend: "host_shell"
    },
  ];
  const tavilyApiKey = options.secrets?.tools?.tavily?.api_key?.trim();
  if (tavilyApiKey) {
    capabilities.push(tavilyWebSearchCapability());
  }
  capabilities.push({
    capability_id: "openclaw.vicuna.ongoing-task.create",
    tool_surface_id: "vicuna.tasks.ongoing.create",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-ongoing-tasks",
    tool_name: "ongoing_task_create",
    tool_family_id: "ongoing_task",
    tool_family_name: "Ongoing Task",
    tool_family_description: "Create or delete one recurring host cron task through a direct tool.",
    method_name: "create",
    method_description: "Create one recurring host cron task for the vicuna runtime.",
    description: "Create one recurring host cron task with explicit text and cadence so the stored prompt runs later as a system message.",
    input_schema_json: {
      type: "object",
      required: ["task_text", "interval", "unit"],
      properties: {
        task_text: {
          type: "string",
          description: "The exact recurring task wording that will later be sent as a system message."
        },
        interval: {
          type: "integer",
          minimum: 1,
          description: "Positive recurrence interval."
        },
        unit: {
          type: "string",
          enum: ["minute", "hour", "day", "week"],
          description: "Cadence unit."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_write",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-ongoing-tasks/tool/ongoing_task_create",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "flattened_ongoing_task_create"
  });
  capabilities.push({
    capability_id: "openclaw.vicuna.ongoing-task.delete",
    tool_surface_id: "vicuna.tasks.ongoing.delete",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-ongoing-tasks",
    tool_name: "ongoing_task_delete",
    tool_family_id: "ongoing_task",
    tool_family_name: "Ongoing Task",
    tool_family_description: "Create or delete one recurring host cron task through a direct tool.",
    method_name: "delete",
    method_description: "Delete one recurring host cron task by task id.",
    description: "Delete one recurring host cron task by task id.",
    input_schema_json: {
      type: "object",
      required: ["task_id"],
      properties: {
        task_id: {
          type: "string",
          description: "Stable recurring cron task identifier."
        }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_write",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-ongoing-tasks/tool/ongoing_task_delete",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "flattened_ongoing_task_delete"
  });
  return assertCapabilityCatalog({
    catalog_version: 1,
    capabilities
  });
}
