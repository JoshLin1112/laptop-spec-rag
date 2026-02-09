"""
速度批次測試模組

此模組用於測量 RAG 系統的 TTFT (首字延遲) 和 TPS (生成速度)。
每一題會記錄這兩個關鍵效能指標。
"""
import csv
import os
import sys
import time
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RAGSystem
from src.prompts import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Paths relative to tests/ directory
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS_FILE = os.path.join(TESTS_DIR, "test_questions.csv")
DEFAULT_OUTPUT_DIR = os.path.join(TESTS_DIR, "results")


def load_questions_from_csv(csv_file):
    """
    從 CSV 檔案載入測試問題
    
    輸入：
        csv_file (str): CSV 檔案路徑
    
    輸出：
        List[dict]: 問題資料列表
    """
    questions = []
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found.")
        return questions
    
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "question": row.get("question", ""),
                "category": row.get("category", "unknown"),
                "expected_answer_type": row.get("expected_answer_type", "unknown")
            })
    
    return questions


def measure_speed_metrics(rag: RAGSystem, user_question: str) -> dict:
    """
    測量單一問題的速度指標
    
    功用：
        對單一問題進行查詢，測量 TTFT 和 TPS。
    
    輸入：
        rag (RAGSystem): RAG 系統實例
        user_question (str): 使用者問題
    
    輸出：
        dict: 包含以下鍵值：
            - ttft: Time To First Token (秒)
            - tps: Tokens Per Second
            - total_tokens: 總生成 token 數
            - total_time: 總生成時間 (秒)
            - answer: 生成的回答
    """
    # Pre-process query (retrieval phase)
    context_str, _ = rag._process_query(user_question)
    prompt = f"Context:\n{context_str}\n\nQuestion: {user_question}"
    
    # Start timing for generation phase
    start_time = time.perf_counter()
    first_token_time = None
    tokens = []
    
    # Use streaming to measure TTFT
    for token in rag.generator.generate_stream(SYSTEM_PROMPT, prompt):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        tokens.append(token)
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_tokens = len(tokens)
    total_time = end_time - start_time
    
    # TTFT: time from start to first token
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    
    # TPS: tokens per second (based on generation time after first token)
    generation_time = end_time - first_token_time if first_token_time else total_time
    tps = total_tokens / generation_time if generation_time > 0 else 0.0
    
    return {
        "ttft": ttft,
        "tps": tps,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "answer": "".join(tokens)
    }


def speed_batch_test(output_file=None, questions_file=None) -> list:
    """
    批次速度測試
    
    功用：
        對所有測試問題進行速度測試，記錄每題的 TTFT 和 TPS。
    
    輸入：
        output_file (str, optional): 輸出 CSV 檔案路徑
        questions_file (str, optional): 問題 CSV 檔案路徑
    
    輸出：
        list: 測試結果列表，每個元素為 dict 包含問題和速度指標
    """
    if questions_file is None:
        questions_file = DEFAULT_QUESTIONS_FILE
    if output_file is None:
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, "speed_results.csv")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("Initializing RAG System...")
    rag = RAGSystem()
    
    questions_data = load_questions_from_csv(questions_file)
    print(f"Loaded {len(questions_data)} questions for speed testing.")
    
    results = []
    
    headers = [
        "Question",
        "Category",
        "TTFT (s)",
        "TPS (tokens/s)",
        "Total Tokens",
        "Total Time (s)"
    ]
    
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i, q_data in enumerate(questions_data):
            question = q_data["question"]
            category = q_data["category"]
            
            print(f"[{i+1}/{len(questions_data)}] [{category}] Testing: {question[:50]}...")
            
            try:
                metrics = measure_speed_metrics(rag, question)
                
                row = [
                    question,
                    category,
                    f"{metrics['ttft']:.4f}",
                    f"{metrics['tps']:.2f}",
                    metrics['total_tokens'],
                    f"{metrics['total_time']:.4f}"
                ]
                writer.writerow(row)
                f.flush()
                
                results.append({
                    "question": question,
                    "category": category,
                    "ttft": metrics['ttft'],
                    "tps": metrics['tps'],
                    "total_tokens": metrics['total_tokens'],
                    "total_time": metrics['total_time']
                })
                
                print(f"    TTFT: {metrics['ttft']:.4f}s | TPS: {metrics['tps']:.2f} | Tokens: {metrics['total_tokens']}")
                
            except Exception as e:
                print(f"Error processing '{question}': {e}")
                writer.writerow([question, category, "ERROR", "ERROR", "ERROR", "ERROR"])
                results.append({
                    "question": question,
                    "category": category,
                    "ttft": None,
                    "tps": None,
                    "total_tokens": None,
                    "total_time": None,
                    "error": str(e)
                })
    
    # Print summary statistics
    valid_results = [r for r in results if r.get('ttft') is not None]
    if valid_results:
        avg_ttft = sum(r['ttft'] for r in valid_results) / len(valid_results)
        avg_tps = sum(r['tps'] for r in valid_results) / len(valid_results)
        
        print(f"\n{'='*60}")
        print(f"Speed Test Summary")
        print(f"{'='*60}")
        print(f"Total Questions: {len(questions_data)}")
        print(f"Successful: {len(valid_results)}")
        print(f"Average TTFT: {avg_ttft:.4f}s")
        print(f"Average TPS: {avg_tps:.2f} tokens/s")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    speed_batch_test()
