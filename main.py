"""
RAG 系統主程式入口

此模組提供命令列互動介面，讓使用者可以透過終端機與 RAG 系統進行對話。
"""
import sys
from src.rag_engine import RAGSystem


def main():
    """
    主程式函數
    
    功用：
        初始化 RAG 系統並啟動互動式命令列介面，
        讓使用者可以輸入問題並獲得 AI 助理的串流回應。
    
    輸入：
        無（從標準輸入讀取使用者問題）
    
    輸出：
        無（將回應輸出至標準輸出）
    
    使用方式：
        執行 python main.py 啟動互動模式，輸入 'exit'、'quit' 或 'q' 退出。
    """
    print("Initialize AI Hardware Assistant (GIGABYTE AORUS MASTER 16 AM6H)...")
    try:
        rag = RAGSystem()
        print("System Ready! (Type 'exit' to keyout)")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            if not user_input.strip():
                continue

            print("Assistant: ", end="", flush=True)
            for token in rag.query(user_input):
                print(token, end="", flush=True)
            print()  # Newline
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
