"""
文本生成模組

此模組提供基於 LLM 的文本生成功能，使用 llama-cpp-python 進行本地推理。
"""
import logging
from typing import Generator as TypeGenerator
from llama_cpp import Llama
import config
import os

logger = logging.getLogger(__name__)


class Generator:
    """
    文本生成器類別
    
    功用：
        載入並使用本地 LLM 模型進行文本生成，支援串流輸出。
    
    屬性：
        model (Llama): llama-cpp-python 的模型實例
    """
    
    def __init__(self, repo_id: str, filename: str):
        """
        初始化生成器
        
        功用：
            從 Hugging Face Hub 載入指定的 GGUF 格式 LLM 模型。
        
        輸入：
            repo_id (str): Hugging Face 倉庫 ID（如 "unsloth/gemma-3-4b-it-GGUF"）
            filename (str): 模型檔案名稱（如 "gemma-3-4b-it-Q4_K_M.gguf"）
        
        輸出：
            無（初始化實例）
        """
        logger.info(f"Loading Generation Model from {repo_id}...")
        import config
        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=getattr(config, "SLM_CONTEXT_LENGTH", 2048),
            cache_type_k="q8_0",
            cache_type_v="q8_0",
            verbose=False
        )

    def generate_stream(self, system_prompt: str, user_prompt: str) -> TypeGenerator[str, None, None]:
        """
        串流式文本生成
        
        功用：
            根據系統提示詞和使用者提示詞，逐 token 生成回應文本。
        
        輸入：
            system_prompt (str): 系統提示詞，定義 AI 助理的角色和行為規則
            user_prompt (str): 使用者提示詞，包含上下文和問題
        
        輸出：
            Generator[str, None, None]: 字串生成器，逐個 yield 生成的 token
        """
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        output = self.model(
            prompt,
            max_tokens=512,
            stop=["<|im_end|>"],
            stream=True,
            temperature=getattr(config, "GENERATION_TEMPERATURE", 0.3)
        )
        
        for chunk in output:
            token = chunk["choices"][0]["text"]
            yield token
