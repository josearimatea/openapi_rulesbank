# src/app/config/logging_config.py
"""
Logging configuration.
Sets up basic logging with console handler.
No logger instance exported—use per-module loggers (as discussed).
Import this in entry points to apply config globally.
"""

import logging
import os
from dotenv import load_dotenv  # Load .env file

load_dotenv()  # Ensure .env is loaded before os.getenv

# LOG_LEVEL Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
# Default usage is INFO or DEBUG for development. Can be set in .env for flexibility.
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
level = getattr(logging, log_level, logging.INFO)  # fallback to INFO if invalid level

# Logging configuration (central and reusable)
logging.basicConfig(
    level=level,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()  # Console output
        # logging.FileHandler("rag_app.log")  # Uncomment for file logging
    ]
)

# Suppress verbose HTTP and SDK debug logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)