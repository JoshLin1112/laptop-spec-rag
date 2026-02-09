# config.py


# 1. LLM Model

# SLM_REPO_ID = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
# SLM_FILENAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
SLM_REPO_ID = "unsloth/gemma-3-4b-it-GGUF"
SLM_FILENAME="gemma-3-4b-it-Q4_K_M.gguf"
SLM_CONTEXT_LENGTH = 4096

# 2. Embedding Model

EMBED_REPO_ID = "Qwen/Qwen3-Embedding-0.6B-GGUF"
EMBED_FILENAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"

# 3. Generation Parameters
GENERATION_TEMPERATURE = 0.7
RETRIEVAL_TOP_K = 5
