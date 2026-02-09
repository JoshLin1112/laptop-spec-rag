"""
RAG 引擎

此檔案實現了完整的 RAG 系統，整合了 Embedding、檢索、生成和後處理等功能。
用於回答 GIGABYTE AORUS MASTER 16 AM6H 系列筆電的硬體規格問題。
"""
import logging
from typing import Generator as TypeGenerator, Tuple, List, Dict
import re
import config
from src.models import Chunk
from src.retriever import EmbeddingEngine, ChunkRetriever, QARetriever
from src.generator import Generator
from src.data_loader import ConvertSpec
from src.query_analyzer import QueryFilter, ProductNameExtractor
from src.prompts import SYSTEM_PROMPT, FALLBACK_RESPONSE_ZH, FALLBACK_RESPONSE_EN

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG 系統主類別
    
    功用：
        整合嵌入引擎、檢索器、生成器和查詢分析器，提供完整的 RAG 問答功能。
        支援產品路由、確定性查找、混合檢索和串流生成。
    
    屬性：
        embed_engine (EmbeddingEngine): 嵌入向量引擎
        index (ChunkRetriever): 區塊檢索器
        generator (Generator): 文本生成器
        loader (ConvertSpec): 規格資料載入器
        chunks (List[Chunk]): 已索引的區塊列表
        query_filter (QueryFilter): 查詢過濾器
        qa_retriever (QARetriever): 問答檢索器
        product_extractor (ProductNameExtractor): 產品名稱提取器
    """
    
    def __init__(self, spec_file: str = "data_processing/knowledge_base/specs_integrated.json"):
        """
        初始化 RAG 系統
        
        功用：
            載入模型、建立索引、初始化所有組件。
        
        輸入：
            spec_file (str): 產品規格 JSON 檔案路徑（預設為 "data_processing/knowledge_base/specs_integrated.json"）
        
        輸出：
            無（初始化實例）
        """
        # Initialize components using parameters from config.py
        self.embed_engine = EmbeddingEngine(
            repo_id=config.EMBED_REPO_ID,
            filename=config.EMBED_FILENAME
        )
        self.index = ChunkRetriever(self.embed_engine)
        self.generator = Generator(
            repo_id=config.SLM_REPO_ID,
            filename=config.SLM_FILENAME
        )
        
        # Load and Index Data
        self.loader = ConvertSpec(spec_file)
        self.chunks = self.loader.chunk_data()
        self.index.add_chunks(self.chunks)
        
        # Initialize Query Filter
        self.query_filter = QueryFilter()
        
        # Initialize QA Retriever for second retrieval path
        self.qa_retriever = QARetriever(self.embed_engine, "data_processing/knowledge_base/qa_dataset.json")
        
        # Initialize Product Name Extractor with dynamic list
        self.product_extractor = ProductNameExtractor(self.loader.get_products())

# =========================================================================
# Core Query Processing (Shared Logic)
# =========================================================================
    
    def _process_query(self, user_question: str) -> Tuple[str, List[Dict]]:
        """
        核心查詢處理（共用邏輯）
        
        功用：
            執行查詢的完整處理流程，包括產品路由、檢索、上下文組裝。
        
        輸入：
            user_question (str): 使用者問題
        
        輸出：
            Tuple[str, List[Dict]]:
                - 組裝好的上下文字串
                - 檢索結果元資料列表，每個元素包含 type、content、score
        """

        extracted_product_id = self.product_extractor.extract(user_question)
        logger.info(f"Router extracted product ID: {extracted_product_id}")
        
        deterministic_facts = []
        retrieval_metadata = []
        
        # 1. Router (Check if it is a "comparison", "specific product", "List products" or "general question")
        is_comparison = self.query_filter.is_comparison_query(user_question)

        if is_comparison:
            # Comparison mode: Directly return ALL Full Specs chunks, skip embedding search
            chunk_results = [c for c in self.chunks if c.metadata.get("key") == "Full Specs"]
            chunk_scores = [1.0] * len(chunk_results)  # Synthetic high score
            logger.info(f"Comparison mode: Returning {len(chunk_results)} Full Specs chunks directly (no search).")

        elif extracted_product_id:
            # Specific product mode: Only look at this product's chunks, then use embedding search
            target_chunks_for_search = [c for c in self.chunks if c.metadata.get("product_id") == extracted_product_id]
            logger.info(f"Strictly filtering to {len(target_chunks_for_search)} chunks for product {extracted_product_id}")
            chunk_results, chunk_scores = self.index.search(user_question, top_k=getattr(config, "RETRIEVAL_TOP_K", 3), candidate_chunks=target_chunks_for_search)
        

        elif self.query_filter.is_product_list_query(user_question):
            # List products mode: Return all products' names + search for general question
            all_products = self.loader.get_products()
            product_names = [p.get("product_name", "Unknown") for p in all_products]
            product_list_str = ", ".join(product_names)
            fact = f"FACT: The available products in the database are: {product_list_str}."
            deterministic_facts.append(fact)
            logger.info(f"Product List detected. Injected: {fact}")
            chunk_results, chunk_scores = self.index.search(user_question, top_k=getattr(config, "RETRIEVAL_TOP_K", 3), candidate_chunks=self.chunks)
        
        else:
            # General question mode: No product extracted, no list intent -> Search all chunks
            chunk_results, chunk_scores = self.index.search(user_question, top_k=getattr(config, "RETRIEVAL_TOP_K", 3), candidate_chunks=self.chunks)
        
        # 2. Dynamic Context Management
        include_full_specs = True
        if deterministic_facts:
            include_full_specs = False
        if chunk_scores and chunk_scores[0] > 1.2:
            include_full_specs = False
        
        # In comparison mode, always include Full Specs
        if is_comparison:
            include_full_specs = True
            
        # Filter out Full Specs from results if needed
        final_chunk_results = []
        final_chunk_scores = []
        for c, s in zip(chunk_results, chunk_scores):
            if not include_full_specs and c.metadata.get("key") == "Full Specs":
                continue
            final_chunk_results.append(c)
            final_chunk_scores.append(s)
            
            # Add to retrieval metadata
            retrieval_metadata.append({
                "type": "chunk",
                "content": c.content,
                "score": float(s)
            })
            
        chunk_context = "\n---\n".join([
            f"{c.content}\n[信心分數: {s:.2f}]" for c, s in zip(final_chunk_results, final_chunk_scores)
        ])
        logger.info(f"Chunk Context:\n{chunk_context}")
        
        # 3. QA Retriever
        qa_product_suffix = "overall"
        if extracted_product_id:
            qa_product_suffix = extracted_product_id.split("-")[-1].upper()
            
        qa_results, qa_scores = self.qa_retriever.search(user_question, qa_product_suffix, top_k=1)
        qa_context = "\n---\n".join([
            f"Q: {qa.question}\nA: {qa.answer}\nScore: {s:.2f}" for qa, s in zip(qa_results, qa_scores)
        ])
        logger.info(f"QA Context:\n{qa_context}")
        
        # Add QA to retrieval metadata
        for qa, s in zip(qa_results, qa_scores):
            retrieval_metadata.append({
                "type": "qa",
                "content": f"Q: {qa.question}\nA: {qa.answer}",
                "score": float(s)
            })

        # 4. Assemble Context
        context_parts = []
        if deterministic_facts:
            context_parts.append("[Verified Facts]\n" + "\n".join(deterministic_facts))
        
        context_parts.append(f"[Retrieval Results]\n{chunk_context}")
        context_parts.append(f"[QA Database]\n{qa_context}")
        
        context_str = "\n\n".join(context_parts)
        
        return context_str, retrieval_metadata

# =========================================================================
# Public Query Methods
# =========================================================================
    
    def query(self, user_question: str) -> TypeGenerator[str, None, None]:
        """
        使用者查詢（串流版本）
        
        功用：
            處理使用者問題並以串流方式返回 AI 回應。
        
        輸入：
            user_question (str): 使用者問題
        
        輸出：
            Generator[str, None, None]: 字串生成器，逐個 yield 生成的 token
        """
        context_str, _ = self._process_query(user_question)
        prompt = f"Context:\n{context_str}\n\nQuestion: {user_question}"
        
        return self.generator.generate_stream(SYSTEM_PROMPT, prompt)
    
    def query_with_metadata(self, user_question: str) -> dict:
        """
        使用者查詢（含元資料版本）
        
        功用：
            處理使用者問題並返回完整回應，包含檢索結果元資料。
            主要用於測試和評估。
        
        輸入：
            user_question (str): 使用者問題
        
        輸出：
            dict: 包含以下鍵值：
                - question: 原始問題
                - retrieval_results: 檢索結果元資料列表
                - answer: 生成的回答
        """
        context_str, retrieval_metadata = self._process_query(user_question)
        prompt = f"Context:\n{context_str}\n\nQuestion: {user_question}"
        # Generate full answer (non-streaming)
        raw_answer = "".join(list(self.generator.generate_stream(SYSTEM_PROMPT, prompt)))
        
        # Post-process the answer (disabled for testing)
        # answer = self._postprocess_answer(raw_answer, user_question)
        answer = raw_answer
        
        return {
            "question": user_question,
            "retrieval_results": retrieval_metadata,
            "answer": answer
        }

    # =========================================================================
    # Answer Post-Processing (Rule-Based Cleanup)
    # =========================================================================
    
    def _postprocess_answer(self, answer: str, question: str) -> str:
        """
        回答後處理（rule-based）
        
        功用：
            對生成的回答進行後處理，移除信心分數、標準化回退訊息。
        
        輸入：
            answer (str): 原始生成的回答
            question (str): 原始問題（用於判斷語言）
        
        輸出：
            str: 清理後的回答
        """
        # Rule 1: Remove confidence score patterns
        score_patterns = [
            r'\[?信心分?數[：:]\s*[\d.]+\]?',  # [信心分數: 0.42] or 信心分數: 0.42
            r'\[?可信度分?數[：:]\s*[\d.]+\]?',  # 可信度分數: 0.42
            r'\[?[Ss]core[：:]\s*[\d.]+\]?',    # Score: 0.42
            r'\[?[Cc]onfidence[：:]\s*[\d.]+\]?',  # Confidence: 0.42
        ]
        
        for pattern in score_patterns:
            answer = re.sub(pattern, '', answer)
        
        # Rule 2: If fallback phrases detected, clean up and return only the fallback
        fallback_indicators_zh = ['抱歉', '沒有這項', '不在我', '無法提供', '沒有相關']
        fallback_indicators_en = ['not available', 'don\'t have', 'cannot provide', 'no information']
        
        # Detect language of question
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in question)
        
        answer_lower = answer.lower()
        
        # Check for fallback indicators
        has_zh_fallback = any(ind in answer for ind in fallback_indicators_zh)
        has_en_fallback = any(ind in answer_lower for ind in fallback_indicators_en)
        
        if has_zh_fallback or has_en_fallback:
            # Return clean fallback response
            if is_chinese:
                return FALLBACK_RESPONSE_ZH
            else:
                return FALLBACK_RESPONSE_EN
        
        # Clean up extra whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer) 
        answer = answer.strip()
        
        return answer
