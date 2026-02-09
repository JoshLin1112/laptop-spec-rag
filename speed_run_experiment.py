"""
速度實驗執行模組

此模組用於執行不同溫度和 top_k 配置的 RAG 系統速度實驗。
將所有實驗結果儲存到單一 Excel 檔案中，每個配置對應一個工作表。
記錄 TTFT (首字延遲) 和 TPS (生成速度) 兩個關鍵指標。
"""
import subprocess
import sys
import os
import time
import csv
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Paths
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
RESULTS_DIR = os.path.join(TESTS_DIR, "results")
QUESTIONS_FILE = os.path.join(TESTS_DIR, "test_questions.csv")
OUTPUT_EXCEL = os.path.join(RESULTS_DIR, "gemma3_speed_results.xlsx")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Suppress logging
logging.basicConfig(level=logging.ERROR)

# Define experiment configurations
experiments = [
    {"temperature": 0.0, "top_k": 3},
    {"temperature": 0.3, "top_k": 3},
    {"temperature": 0.7, "top_k": 3},
    {"temperature": 0.0, "top_k": 5},
    {"temperature": 0.3, "top_k": 5},
    {"temperature": 0.7, "top_k": 5},
]


def update_config(temperature: float, top_k: int):
    """
    更新配置檔案
    
    功用：
        根據指定的參數更新 config.py 檔案，設定新的溫度和 top_k 值。
    
    輸入：
        temperature (float): 生成溫度（0.0 - 1.0）
        top_k (int): 檢索返回的最相關區塊數量
    
    輸出：
        無（寫入 config.py 檔案）
    """
    config_content = f'''# config.py


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
GENERATION_TEMPERATURE = {temperature}
RETRIEVAL_TOP_K = {top_k}
'''
    with open(os.path.join(PROJECT_ROOT, "config.py"), "w", encoding="utf-8") as f:
        f.write(config_content)
    print(f"Updated config: TEMPERATURE={temperature}, TOP_K={top_k}")


def load_questions():
    """載入測試問題"""
    questions = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "question": row.get("question", ""),
                "category": row.get("category", "unknown"),
            })
    return questions


def run_speed_test(top_k: int) -> tuple:
    """
    執行速度測試
    
    功用：
        載入測試問題，使用 RAG 系統回答所有問題，並記錄 TTFT 和 TPS。
    
    輸入：
        top_k (int): 檢索返回的最相關區塊數量
    
    輸出：
        tuple: (rows, headers, summary)
            - rows (List[List]): 測試結果列表
            - headers (List[str]): 欄位標題列表
            - summary (dict): 摘要統計資料
    """
    import importlib
    import config
    importlib.reload(config)
    
    # Reload rag_engine to use new config
    from src import rag_engine
    importlib.reload(rag_engine)
    
    from src.rag_engine import RAGSystem
    from src.prompts import SYSTEM_PROMPT
    
    print(f"Initializing RAG System...")
    rag = RAGSystem()
    
    # Load questions
    questions = load_questions()
    print(f"Loaded {len(questions)} questions.")
    
    headers = ["Question", "Category", "TTFT (s)", "TPS (tokens/s)", "Total Tokens", "Total Time (s)"]
    rows = []
    
    ttft_list = []
    tps_list = []
    
    for i, q_data in enumerate(questions):
        question = q_data["question"]
        category = q_data["category"]
        
        print(f"[{i+1}/{len(questions)}] [{category}] Testing: {question[:50]}...")
        
        try:
            # Pre-process query (retrieval phase)
            context_str, _ = rag._process_query(question)
            prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
            
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
            
            row = [
                question,
                category,
                round(ttft, 4),
                round(tps, 2),
                total_tokens,
                round(total_time, 4)
            ]
            rows.append(row)
            
            ttft_list.append(ttft)
            tps_list.append(tps)
            
            print(f"    TTFT: {ttft:.4f}s | TPS: {tps:.2f}")
            
        except Exception as e:
            print(f"Error processing '{question}': {e}")
            rows.append([question, category, "ERROR", "ERROR", "ERROR", "ERROR"])
    
    # Calculate summary statistics
    summary = {
        "avg_ttft": sum(ttft_list) / len(ttft_list) if ttft_list else 0,
        "avg_tps": sum(tps_list) / len(tps_list) if tps_list else 0,
        "min_ttft": min(ttft_list) if ttft_list else 0,
        "max_ttft": max(ttft_list) if ttft_list else 0,
        "min_tps": min(tps_list) if tps_list else 0,
        "max_tps": max(tps_list) if tps_list else 0,
        "total_questions": len(questions)
    }
    
    print(f"\nAverage TTFT: {summary['avg_ttft']:.4f}s | Average TPS: {summary['avg_tps']:.2f}\n")
    
    return rows, headers, summary


def main():
    """
    主程式函數
    
    功用：
        執行所有定義的實驗配置，將速度測試結果儲存到 Excel 檔案中。
        每個配置的結果儲存在獨立的工作表，並生成摘要工作表。
    
    輸入：
        無
    
    輸出：
        無（生成 Excel 結果檔案並輸出執行時間摘要）
    """
    print("=" * 60)
    print("Running Speed Experiments with Different Configurations")
    print(f"Results will be saved to: {OUTPUT_EXCEL}")
    print("=" * 60)
    
    # Import pandas for Excel writing
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas openpyxl")
        sys.exit(1)
    
    # Store experiment results
    all_results = {}  # sheet_name -> DataFrame
    experiment_summaries = []
    
    for i, exp in enumerate(experiments):
        temp = exp["temperature"]
        top_k = exp["top_k"]
        sheet_name = f"K{top_k}_T{temp}"
        
        print(f"\n[Experiment {i+1}/6] Temperature={temp}, Top_K={top_k}")
        print("-" * 40)
        
        # Update config
        update_config(temp, top_k)
        
        # Run speed test and get results
        rows, headers, summary = run_speed_test(top_k)
        
        # Create DataFrame for this experiment
        df = pd.DataFrame(rows, columns=headers)
        all_results[sheet_name] = df
        
        experiment_summaries.append({
            "experiment": i + 1,
            "temperature": temp,
            "top_k": top_k,
            "avg_ttft": summary["avg_ttft"],
            "avg_tps": summary["avg_tps"],
            "min_ttft": summary["min_ttft"],
            "max_ttft": summary["max_ttft"],
            "min_tps": summary["min_tps"],
            "max_tps": summary["max_tps"]
        })
    
    # Write all results to a single Excel file with multiple sheets
    print(f"\nWriting results to {OUTPUT_EXCEL}...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # Write detailed results for each experiment
        for sheet_name, df in all_results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Add summary sheet
        summary_data = []
        for exp_result in experiment_summaries:
            summary_data.append({
                "Experiment": exp_result["experiment"],
                "Temperature": exp_result["temperature"],
                "Top_K": exp_result["top_k"],
                "Avg TTFT (s)": round(exp_result["avg_ttft"], 4),
                "Avg TPS (tokens/s)": round(exp_result["avg_tps"], 2),
                "Min TTFT (s)": round(exp_result["min_ttft"], 4),
                "Max TTFT (s)": round(exp_result["max_ttft"], 4),
                "Min TPS": round(exp_result["min_tps"], 2),
                "Max TPS": round(exp_result["max_tps"], 2)
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    print("=" * 60)
    print("All experiments completed!")
    print("=" * 60)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Speed Experiment Summary (TTFT & TPS)")
    print("=" * 80)
    print(f"{'Exp':<5} {'Temp':<6} {'K':<4} {'Avg TTFT (s)':<14} {'Avg TPS':<12}")
    print("-" * 80)
    for exp_result in experiment_summaries:
        print(f"{exp_result['experiment']:<5} "
              f"{exp_result['temperature']:<6} "
              f"{exp_result['top_k']:<4} "
              f"{exp_result['avg_ttft']:<14.4f} "
              f"{exp_result['avg_tps']:<12.2f}")
    print("-" * 80)
    print(f"Results saved to: {OUTPUT_EXCEL}")
    print("=" * 80)


if __name__ == "__main__":
    main()
