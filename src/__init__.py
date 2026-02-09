"""
RAG 系統核心模組

此套件包含 RAG 系統的所有核心元件：
- models: 資料模型 (Chunk, QAItem)
- data_loader: 資料載入與轉換
- retriever: 嵌入引擎與檢索器
- generator: 文本生成器
- query_analyzer: 查詢分析器
- rag_engine: RAG 系統主引擎
- prompts: 系統提示詞
"""

from src.models import Chunk, QAItem
from src.rag_engine import RAGSystem

__all__ = ["Chunk", "QAItem", "RAGSystem"]
