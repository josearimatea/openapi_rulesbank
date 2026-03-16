# src/agents/reader_agent.py
# Reader Agent: Loads and parses the main 3GPP document (DOC_TEST) and helper docs (DOC_HELPER).
# Focuses on extracting relevant sections for rule extraction (e.g., templates, NRM, mappings to JSON/YANG/OpenAPI).
# Uses deterministic parsing first, then optional LLM for summarization/filtering.
# Output: Dict with 'parsed_main' (list of relevant sections from main doc) and 'helper_context' (summaries from helpers).

import markdown
import requests  # For fetching URLs (uv add requests)
from config import get_logger, get_langchain_llm, PROMPTS_DIR, SETTINGS
from config.settings import DOC_TEST, DOC_HELPER  # From config/settings.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = get_logger(__name__)

def load_markdown(file_path: str) -> str:
    """Loads Markdown content from local file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

def fetch_url(url: str) -> str:
    """Fetches content from URL (e.g., OpenAPI specs)."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        raise

def parse_sections(md_text: str) -> list:
    """Parses Markdown into sections (split by ## headers)."""
    sections = [section.strip() for section in md_text.split('## ') if section.strip()]
    # Filter relevant: Keywords for 3GPP/OpenAPI rules (templates, mappings, stages)
    relevant_keywords = ['template', 'nrm', 'mapping', 'stage', 'json', 'yang', 'openapi', 'attribute', 'class']
    relevant_sections = [s for s in sections if any(kw in s.lower() for kw in relevant_keywords)]
    return relevant_sections

def reader_agent(state: dict, llm) -> dict:
    """Main agent function: Processes inputs and updates state."""
    logger.info("Starting Reader Agent.")

    # Load and parse main document (DOC_TEST)
    main_md = load_markdown(DOC_TEST) if not DOC_TEST.startswith('http') else fetch_url(DOC_TEST)
    main_sections = parse_sections(main_md)
    logger.debug(f"Parsed {len(main_sections)} relevant sections from main doc.")

    # Optional LLM summarization for main doc (if prompt exists)
    prompt_path = PROMPTS_DIR / "reader.txt"  # Content: "Summarize relevant sections from {text} focusing on 3GPP rules for OpenAPI..."
    if prompt_path.exists() and SETTINGS.DEBUG:  # Only if debug mode
        with open(prompt_path, 'r') as p:
            prompt_text = p.read()
        chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_text))
        summarized_main = chain.run({"text": "\n".join(main_sections)})
        state["parsed_main"] = summarized_main
    else:
        state["parsed_main"] = "\n".join(main_sections)

    # Load helpers (DOC_HELPER)
    helper_context = {}
    for name, path in DOC_HELPER.items():
        if path.startswith('http'):
            content = fetch_url(path)
        else:
            content = load_markdown(path)
        # Simple summary (no full parse for helpers)
        helper_context[name] = content[:2000]  # Truncate for efficiency; expand if needed
    state["helper_context"] = helper_context
    logger.debug("Loaded helper contexts.")

    logger.info("Reader Agent complete.")
    return state

# For standalone testing
if __name__ == "__main__":
    from config import get_langchain_llm
    llm = get_langchain_llm()
    test_state = {}
    updated_state = reader_agent(test_state, llm)
    print(updated_state)