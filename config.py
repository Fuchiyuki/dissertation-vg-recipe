# OpenAI API Configuration
OPENAI_API_KEY = "your-openai-api-key-here"

# Model Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"  # set to this model for now since I do not want to spend a lot of money
TEMPERATURE = 0.0  # Set to 0 for reproducibility

# Experiment Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Knowledge Base Configuration
KNOWLEDGE_SOURCES = {
    "cooking_techniques": True,
    "ingredient_properties": True,
    "cooking_commonsense": True
}

# Chain-of-Knowledge Configuration
COK_CONFIG = {
    "consistency_threshold": 0.6,  # Threshold for self-consistency check
    "num_samples": 5,  # Number of samples for self-consistency
    "max_knowledge_items": 5,  # Maximum knowledge items to use per step
    "enable_progressive_correction": True  # Enable progressive rationale correction
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "compute_bertscore": True,  # BERTScore is computationally expensive
    "bertscore_model": "microsoft/deberta-base-mnli",
    "save_intermediate_results": True
}

# Paths
DATA_PATH = "data/pizzacommonsense"
RESULTS_PATH = "results"
CACHE_PATH = "cache"  # For caching API responses

# API Rate Limiting
RATE_LIMIT_DELAY = 0.5  # Delay between API calls in seconds
MAX_TOKENS_PER_REQUEST = 300

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"