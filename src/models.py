"""
RAG 系統資料模型

此模組定義了 RAG 系統中使用的核心資料結構，包括文本區塊和問答項目。
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class Chunk:
    """
    文本區塊類別
    
    功用：
        表示一個文本區塊及其相關的元資料，用於儲存產品規格資訊。
    
    屬性：
        content (str): 區塊的文本內容，包含產品規格資訊
        metadata (Dict[str, str]): 元資料字典，包含 product_id、product_name、category、key 等資訊
    """
    content: str
    metadata: Dict[str, str]


@dataclass
class QAItem:
    """
    問答項目類別
    
    功用：
        表示一組問答對，用於 QA 檢索器中的問答資料集。
    
    屬性：
        question (str): 問題文本
        answer (str): 答案文本
        product_name (str): 相關產品名稱（如 "BXH"、"BYH"、"overall"）
    """
    question: str
    answer: str
    product_name: str
