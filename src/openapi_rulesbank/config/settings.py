# src/app/config/settings.py
"""
Settings values and environment variables.
Centralizes paths, hosts, ports, and API keys.
No side effects on import.
"""

import os
from .paths import *  # Import all paths and settings from paths.py for centralization
from dotenv import load_dotenv  # Load .env file
from qdrant_3gpp_rag.embedding_config import DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL

load_dotenv()  # Ensure .env is loaded before os.getenv

# APP_ENV is set in .env and determines which dataset and collection to use
# APP_ENV Options: "prod", "test" or "test_file"
APP_ENV = os.getenv("APP_ENV", "test_file")  # Default to 'test_file' for safety

# TSPEC_DATA settings
TSPEC_DATA = TSPEC_DATA_TEST_FILE if APP_ENV == 'test_file' else TSPEC_DATA_TEST if APP_ENV == 'test' else TSPEC_DATA_PROD

# Chunk settings
CHUNKS_FILE = CHUNKS_FILE_TEST_FILE if APP_ENV == 'test_file' else CHUNKS_FILE_TEST if APP_ENV == 'test' else CHUNKS_FILE_PROD

# Qdrant settings
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
COLLECTION_NAME = COLLECTION_NAME_TEST_FILE if APP_ENV == 'test_file' else COLLECTION_NAME_TEST if APP_ENV == 'test' else COLLECTION_NAME_PROD

# Retrieval settings
NUMBER_RETRIEVE_CHUNKS = 10

# LLM provider — "openrouter" | "openai".
# Both speak the OpenAI chat-completions protocol, so the same ChatOpenAI client
# serves either one; the provider only decides which key, base URL and headers it
# is built with. The key is validated in llm_config for the selected provider, so
# only the provider actually in use needs a key in .env.
PROVIDER = os.getenv("RULESBANK_PROVIDER", "openrouter").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# OpenRouter attributes usage to an app when these headers are sent; both optional.
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "openapi-rulesbank")

# LLM Settings
# Each provider names its model differently — OpenRouter needs the vendor prefix,
# OpenAI takes the bare name — and switching RULESBANK_PROVIDER by hand without
# switching the model fails at the first call with model_not_found. So the default
# follows the selected provider, exactly as the API key does.
# RULESBANK_MODEL still overrides the default when a run needs another model.
# Measured 15/set/2026 on the phase 2 anchor prompt, same prompt and schema: the
# model name makes no difference here, and neither does the provider. What decides
# is whether reasoning is switched OFF (see REASONING_EFFORT below): with
# reasoning disabled both "openai/gpt-5.6-luna" and "-pro" answered wrongly every
# time; with it left alone both answered correctly every time, as does OpenAI
# direct — where "none" happens to omit the parameter rather than disable it.
DEFAULT_MODEL_BY_PROVIDER = {
    "openrouter": "openai/gpt-5.6-luna",
    # "openrouter": "openai/gpt-5.6-luna-pro",
    "openai":     "gpt-5.6-luna",
}
MODEL = (
    os.getenv("RULESBANK_MODEL")
    or DEFAULT_MODEL_BY_PROVIDER.get(PROVIDER, "gpt-5.6-luna")
)
TEMPERATURE = float(os.getenv("RULESBANK_TEMPERATURE", "0"))
# Reasoning models reject function-calling tools on /v1/chat/completions unless
# reasoning is switched off, and every node here uses structured output via
# function calling. "none" turns reasoning off; low | medium | high enable it;
# empty omits the parameter, which is what models without a reasoning mode expect.
REASONING_EFFORT = os.getenv("RULESBANK_REASONING_EFFORT", "none").strip().lower()
# Per-request timeout (seconds) and automatic retries for the OpenAI client.
# Without a timeout the SDK waits on a hung request for its long default, which
# stalled a whole run for ~6 minutes on a single Phase-2/3 call. A short timeout
# makes a hung call fail fast so max_retries can reissue it instead.
LLM_REQUEST_TIMEOUT = 60
LLM_MAX_RETRIES = 2

# OpenAPI reference version to fetch.
# The 3GPP ProvMnS specs under data/inputs/3gpp declare `openapi: 3.0.1`, so the
# reference used to generate rules must match that legacy version rather than the
# latest published spec (e.g. 3.2.0).
OPENAPI_REFERENCE_VERSION = "3.0.1"

# RAG — OpenAPI reference collection
# The embedding model comes from qdrant_3gpp_rag, the single source for every
# repository: queries must use the model the shared collections were built with.
OPENAPI_COLLECTION_NAME  = "openapi_reference"
CHUNK_SIZE               = 1000  # characters per chunk
CHUNK_OVERLAP            = 200   # overlap between consecutive chunks
EMBEDDING_MODEL          = DEFAULT_EMBEDDING_MODEL
EMBEDDING_DIM            = DEFAULT_EMBEDDING_DIM
OPENAPI_REFERENCE_RETRIEVE_CHUNKS = 5  # number of chunks returned by search_openapi_reference()

# Spec RAG — the main 3GPP document is indexed into a temporary Qdrant collection
# per run (via qdrant_3gpp_rag) so the Extractor can retrieve related sections
# (e.g. the stage 2 definition behind a stage 3 mapping). Dropped by the Builder.
SPEC_COLLECTION_PREFIX  = "spec_tmp_"  # temporary collection name prefix (+ random suffix)
SPEC_RETRIEVE_CHUNKS    = 4            # chunks for the stage 2 canonical-name lookup
# The stage 3 lookup must surface the sibling resource AND type sections of the
# service (path, schema name/casing). They compete with same-service notification
# and overview sections, so it needs a wider net than the focused stage 2 lookup;
# 8 was the point where the resource and type sections reliably came in.
SPEC_SIBLING_RETRIEVE_CHUNKS = 8       # chunks for the stage 3 sibling lookup
# One document section should stay one chunk so a retrieved type/resource carries
# its heading (name/casing) and all its rows together, not a nameless table
# fragment. At 1000 chars, 28% of sections were split; at 3000, only ~10% are —
# the oversized data tables, which the Extractor rarely needs. Larger than the
# global CHUNK_SIZE (which serves the reference/corpus collections) on purpose.
SPEC_CHUNK_SIZE         = 3000         # chunk size for the temporary spec RAG
SPEC_CHUNK_OVERLAP      = 200          # overlap for the temporary spec RAG

# 3GPP corpus RAG — a persistent Qdrant collection holding many 3GPP specs
# (built once via qdrant_3gpp_rag/scripts/index_3gpp_all.py). The ContractBuilder
# and Extractor query it to ground rules in referenced specs (e.g. the MnS
# definitions in TS 28.533/32.160, or the NotificationHeader in TS 28.623). This
# project only reads the collection.
# Candidates are tried in order; the first one that exists in Qdrant is used. This
# lets a full-release index ("...all") take precedence when present, while falling
# back to the 28-series subset that is enough for management-service specs.
GPP_CORPUS_COLLECTIONS  = ["3gpp_rel18_all", "3gpp_rel18_28"]
GPP_CORPUS_RETRIEVE_CHUNKS = 5              # chunks returned by search_3gpp_corpus()
# Looking up an imported common type's definition needs a wider net: the corpus
# was chunked small, so a type's fields (e.g. NotificationHeader's href,
# notificationId, eventTime, systemDN) are spread across several chunks. At k=5
# they don't all come back; ~20 gathers the whole field set even though the
# fields sit in different chunks. Used by search_imported_type_ref().
IMPORTED_TYPE_RETRIEVE_CHUNKS = 20

# Validated-rules RAG — rules are indexed into a temporary collection as they are
# validated, so the Validator can spot a rule that duplicates one already accepted
# in ANOTHER section (the per-section check alone never sees those). Dropped by
# the Builder at the end of the run.
RULES_COLLECTION_PREFIX = "rules_tmp_"  # temporary collection name prefix (+ random suffix)
RULES_RETRIEVE_CHUNKS   = 5             # similar rules shown to the Validator per rule
# The Extractor queries the same rules RAG with a whole section (not one rule), to
# see which of the section's rules already exist elsewhere and avoid recreating
# them. It needs a wider net than the Validator's per-rule check — a section maps
# to many rules — so it retrieves more; still k-bounded, so context stays flat
# even for large services (e.g. ProvMnS ~200 rules total).
RULES_EXTRACTOR_RETRIEVE_CHUNKS = 15    # existing rules shown to the Extractor per section

# Pipeline flow control
# Missing-context retrieval — when the Extractor reports a fact it needed and
# could not find, the Reflector turns the gap into one query and the answer is
# handed to the section's next attempt. Two attempts can carry retrieved context
# (attempts 2 and 3 of MAX_ITERATIONS), which is what bounds how often this runs.
# One query per gap, and the Extractor reports at most 2 gaps, so a section adds
# at most 2 retrievals per attempt.
MAX_CONTEXT_QUERIES     = 2   # gap queries run per section per attempt
# Each query searches all three collections — this document, the 3GPP corpus and
# the OpenAPI reference — rather than one the Reflector picks. Routing proved
# brittle in both directions: a media-type gap sent only to the OpenAPI reference
# came back defining what a media type is and never which one to use, and a gap
# about a field defined in another spec but USED in a clause of this one went to
# the corpus, so the clause holding the answer was never searched. Three lookups
# per gap cost little and remove the decision; the chunk count per collection is
# lower to match, keeping what reaches the prompt about the same.
CONTEXT_RETRIEVE_CHUNKS = 4   # chunks reaching the prompt per collection searched

VALIDATION_ERROR_THRESHOLD = 0.10  # Loop back to Extractor if error rate exceeds 10%
MAX_ITERATIONS = 3                  # Maximum Extractor → Validator loops before forcing forward

# ContractBuilder — semantic scope contract for the target service
# CONTRACT_LOCATOR_PREVIEW_CHARS: chars of each section's content sent in the
#   locator table (LLM call 1). The locator only needs a preview to decide
#   relevance — the full content of located sections goes to LLM call 2.
# CONTRACT_FALLBACK_SECTIONS: sections used when the locator finds nothing — the
#   opening sections of a 3GPP spec are usually the overview and definitions.
CONTRACT_LOCATOR_PREVIEW_CHARS = 600
CONTRACT_FALLBACK_SECTIONS     = 5

# Section parsing — title validity filter
# When True, parse_sections() discards sections whose title contains no real words
# (i.e., titles made up entirely of symbols like table borders: "+---+---+").
# This safely removes 3GPP document cover-page tables that pass the keyword filter
# because their cell content contains relevant keywords (e.g. "NRM", "mapping").
# Set to False to disable and keep all sections regardless of title format.
FILTER_SYMBOLIC_TITLES = True

# Section parsing — keyword relevance filter
# When True, parse_sections() keeps only sections whose text contains at least one
# keyword from RELEVANT_KEYWORDS in parsers.py. Disabled by default because the
# keyword list may not cover all relevant terminology in a given spec.
# Enable only after reviewing and validating the keyword list for the target document.
FILTER_BY_KEYWORDS = False

# Target OpenAPI YAML — which OpenAPI document the rules are built for. Used as a
# label for the legacy YAML and downstream metadata; the rules are built from the
# spec sections only (the document's Annex A is not used as context).
#
# In normal use the target is REQUIRED from the user via main.py --target. This
# value is the fallback used by local tests and programmatic reader_node() calls
# that don't set state["target_yaml"].
#
# It is the single switch for local testing: change it to run the notebooks and
# the integration tests against another Management Service in the same document
# (e.g. "PerfMnS", "FaultMnS"). Outputs follow it into that service's folder.
TARGET_YAML = "ProvMnS"

# Optional legacy OpenAPI YAML — a previous release's YAML (e.g. Rel-17) given as
# extra context for building Rel-18 rules. Provided via main.py --legacy-yaml; no
# default (the feature is off unless a path is passed). Truncated by the value
# below when injected into helper_context.
LEGACY_YAML_MAX_CHARS = 8000

# Diagnostic mode — when True, the Validator records per-iteration details
# (reflected rules, validation errors, section feedback, error rate) and the
# Builder writes a full JSON report to data/outputs/diagnostic/.
# Set to False in production to avoid the overhead.
DIAGNOSTIC_MODE = False