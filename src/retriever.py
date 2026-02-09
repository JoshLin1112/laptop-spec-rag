"""
檢索器模組

此模組提供向量嵌入引擎和檢索器功能，用於從產品規格 chunk 和問答資料集中檢索相關資訊。
包含 EmbeddingEngine（嵌入引擎）、ChunkRetriever（ chunk 檢索器）和 QARetriever（問答檢索器）。
"""
import json
import logging
import numpy as np
from typing import List
import os
from llama_cpp import Llama

# Support both direct execution and module import
try:
    from src.models import Chunk, QAItem
except ImportError:
    from models import Chunk, QAItem

# Configure Logging
logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    功用：
        載入並使用本地嵌入模型將文本轉換為向量表示。
    """
    
    def __init__(self, repo_id: str, filename: str):
        """
        從 Hugging Face Hub 載入指定的 GGUF 格式 Embedding Model。
        
        輸入：
            repo_id (str): Hugging Face 倉庫 ID（如 "Qwen/Qwen3-Embedding-0.6B-GGUF"）
            filename (str): 模型檔案名稱（如 "Qwen3-Embedding-0.6B-Q8_0.gguf"）
        """
        logger.info(f"Loading Embedding Model from {repo_id}...")
        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            embedding=True,
            verbose=False
        )

    def embed(self, text: str) -> np.ndarray:
        """
        將文本轉換為 Embedding 
        
        功用：
            使用 Embedding Model 將輸入文本轉換為向量。
        
        輸入：
            text (str): 要轉換的文本字串
        
        輸出：
            np.ndarray: 文本的 Embedding 
        """
        # llama-cpp-python returns a list of floats
        embed_list = self.model.create_embedding(text)["data"][0]["embedding"]
        return np.array(embed_list, dtype=np.float32)


# For Product Spec Dataset
# ==============================
# Chunk Example : 
# Product: AORUS MASTER 16 BYH
# Category: Processor
# Specification: OS
# Value: Windows 11 Pro (GIGABYTE recommends Windows 11 Pro for business.)
# Windows 11 Home
# UEFI Shell OS
# ==============================
class ChunkRetriever:
    """
     chunk 檢索器類別
    
    功用：
        使用混合搜尋（餘弦相似度 + 關鍵字匹配）從產品規格 chunk 中檢索相關資訊。
    
    屬性：
        engine (EmbeddingEngine): 嵌入引擎實例
        chunks (List[Chunk]): 已索引的 chunk 列表
        vectors (np.ndarray): Embedding 矩陣（已正規化）
    """
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        """
        建立檢索器實例並關聯嵌入引擎。
        
        輸入：
            embedding_engine (EmbeddingEngine): 用於生成 Embedding 的引擎實例

        """
        self.engine = embedding_engine
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = None

    def add_chunks(self, chunks: List[Chunk]):
        """
        功用：
            將 chunk 列表加入索引，為每個 chunk 生成 Embedding 並正規化。
        
        輸入：
            chunks (List[Chunk]): 要索引的 chunk 列表
        
        輸出：
            無（更新內部 chunks 和 vectors 屬性）
        """
        # Embed all chunks
        self.chunks = chunks
        vectors_list = []
        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                logger.info(f"Embedding chunk {i+1}/{len(chunks)}")
            vec = self.engine.embed(chunk.content)
            vectors_list.append(vec)
        
        self.vectors = np.stack(vectors_list)
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def search(self, query: str, top_k: int = 3, candidate_chunks: List[Chunk] = None) -> List[Chunk]:
        """
        混合搜尋相關 chunk 
        
        功用：
            使用向量相似度和關鍵字匹配的混合搜尋，
            從候選 chunk 中找出與查詢最相關的 top_k 個 chunk 。
        
        輸入：
            query (str): 使用者查詢文本
            top_k (int): 返回的最相關 chunk 數量（預設為 3）
            candidate_chunks (List[Chunk]): 候選 chunk 列表（可選，預設搜尋所有 chunk ）
        
        輸出：
            Tuple[List[Chunk], List[float]]: 
                - 相關 chunk 列表（按相關性降序排列）
                - 對應的相似度分數列表
        """
        # 1. Vector Search Preparation
        query_vec = self.engine.embed(query)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        
        # 2. Keyword Search Preparation
        query_tokens = set(query.lower().split())
        
        target_chunks = candidate_chunks if candidate_chunks is not None else self.chunks
        if not target_chunks:
            return [], []

        target_indices = []
        target_vectors_list = []
        
        for t_chunk in target_chunks:
            try:
                idx = self.chunks.index(t_chunk)
                target_indices.append(idx)
                target_vectors_list.append(self.vectors[idx])
            except ValueError:
                continue
                
        if not target_vectors_list:
            return [], []
            
        target_vectors = np.stack(target_vectors_list)
        
        # 3. Compute Vector Scores
        vector_scores = np.dot(target_vectors, query_vec)
        
        # 4. Compute Keyword Scores & Hybrid Fuse
        final_scores = []
        for i, score in enumerate(vector_scores):
            chunk = target_chunks[i]
            content_lower = chunk.content.lower()
            
            overlap = sum(1 for token in query_tokens if token in content_lower)
            keyword_score = min(overlap * 0.05, 0.5) 
            
            hybrid_score = score + keyword_score
            final_scores.append(hybrid_score)
            
        final_scores = np.array(final_scores)
        
        # 5. Get Top-K
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = [target_chunks[i] for i in top_indices]
        score_list = [final_scores[i] for i in top_indices]
        
        return results, score_list


# For QA Dataset
class QARetriever:
    """
    問答檢索器類別
    
    功用：
        從問答資料集中檢索與使用者查詢最相關的問答對，
        支援按產品名稱過濾。
    
    屬性：
        engine (EmbeddingEngine): 嵌入引擎實例
        qa_items (List[QAItem]): 已載入的問答項目列表
        vectors (np.ndarray): 問題的 Embedding 矩陣（已正規化）
    """
    
    def __init__(self, embedding_engine: EmbeddingEngine, qa_file: str = "data_processing/knowledge_base/qa_dataset.json"):
        """
        功用：
            建立檢索器實例，載入問答資料集並生成 Embedding 。
        
        輸入：
            embedding_engine (EmbeddingEngine): 用於生成 Embedding 的引擎實例
            qa_file (str): 問答資料集 JSON 檔案路徑（預設為 "data_processing/knowledge_base/qa_dataset.json"）
        """
        self.engine = embedding_engine
        self.qa_items: List[QAItem] = []
        self.vectors: np.ndarray = None
        self._load_qa_dataset(qa_file)
    
    def _load_qa_dataset(self, qa_file: str):
        """
        載入問答資料集
        
        功用：
            從 JSON 檔案載入問答對，並為每個問題生成 Embedding 。
        
        輸入：
            qa_file (str): 問答資料集 JSON 檔案路徑
        
        輸出：
            無（更新內部 qa_items 和 vectors 屬性）
        """
        try:
            with open(qa_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            qa_pairs = data.get("qa_pairs", [])
            vectors_list = []
            
            for i, qa in enumerate(qa_pairs):
                item = QAItem(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    product_name=qa.get("product_name", "overall")
                )
                self.qa_items.append(item)
                
                vec = self.engine.embed(item.question)
                vectors_list.append(vec)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Embedding QA question {i+1}/{len(qa_pairs)}")
            
            if vectors_list:
                self.vectors = np.stack(vectors_list)
                norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
                self.vectors = self.vectors / (norm + 1e-10)
                logger.info(f"Loaded {len(self.qa_items)} QA pairs from {qa_file}")
            
        except FileNotFoundError:
            logger.warning(f"QA dataset file not found: {qa_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing QA dataset: {e}")
    
    def search(self, query: str, product_name: str = "overall", top_k: int = 2) -> List[QAItem]:
        """
        搜尋相關問答對
        
        功用：
            根據查詢文本和產品名稱，從問答資料集中檢索最相關的問答對。
        
        輸入：
            query (str): 使用者查詢文本
            product_name (str): 產品名稱過濾（如 "BXH"、"overall"）（預設為 "overall"）
            top_k (int): 返回的最相關問答對數量（預設為 2）
        
        輸出：
            Tuple[List[QAItem], List[float]]: 
                - 相關問答對列表（按相關性降序排列）
                - 對應的相似度分數列表
        """
        if not self.qa_items or self.vectors is None:
            return [], []
            
        if product_name is None:
            product_name = "overall"
        
        # 1. Filter by product_name (include both specific model and "overall")
        candidate_indices = []
        for i, qa in enumerate(self.qa_items):
            p_name = qa.product_name if qa.product_name else "overall"
            if p_name.upper() == product_name.upper() or p_name == "overall":
                candidate_indices.append(i)
        
        # 2. If no candidates found, use all items
        if not candidate_indices:
            candidate_indices = list(range(len(self.qa_items)))
        
        # 3. Embed query and normalize
        query_vec = self.engine.embed(query)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        # 4. Compute scores only for candidates
        candidate_vectors = self.vectors[candidate_indices]
        scores = np.dot(candidate_vectors, query_vec)

        # 5. Get top-k from candidates
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        results = [self.qa_items[candidate_indices[i]] for i in sorted_indices]
        score_list = [scores[i] for i in sorted_indices]
        
        return results, score_list
