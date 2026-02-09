import csv
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RAGSystem

# Configure logging
logging.basicConfig(level=logging.ERROR) 
logger = logging.getLogger(__name__)

# Paths relative to tests/ directory
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS_FILE = os.path.join(TESTS_DIR, "test_questions.csv")
DEFAULT_OUTPUT_DIR = os.path.join(TESTS_DIR, "results")

def load_questions_from_csv(csv_file):
    """Load questions from test_questions.csv"""
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

def batch_test(output_file=None, questions_file=None):
    if questions_file is None:
        questions_file = DEFAULT_QUESTIONS_FILE
    if output_file is None:
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, "test_results.csv")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("Initializing RAG System...")
    rag = RAGSystem()
    
    questions_data = load_questions_from_csv(questions_file)
    print(f"Loaded {len(questions_data)} questions for batch testing.")
    
    headers = [
        "Question",
        "Category",
        "Expected Type",
        "Retrieval 1",
        "Retrieval 2",
        "Retrieval 3",
        "Generated Answer"
    ]
    
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i, q_data in enumerate(questions_data):
            question = q_data["question"]
            category = q_data["category"]
            expected_type = q_data["expected_answer_type"]
            
            print(f"[{i+1}/{len(questions_data)}] [{category}] Processing: {question}")
            try:
                result = rag.query_with_metadata(question)
                
                # Format retrieval results for CSV (now supports 3 retrievals)
                metadata = result["retrieval_results"]
                retrieval_cols = []
                for j in range(3):
                    if j < len(metadata):
                        item = metadata[j]
                        content = item["content"].replace("\n", " ")[:200]  # Truncate for readability
                        score = f"{item['score']:.4f}"
                        retrieval_cols.append(f"[{item['type'].upper()}][{score}] {content}")
                    else:
                        retrieval_cols.append("N/A")
                
                # Write row
                row = [
                    question,
                    category,
                    expected_type,
                    retrieval_cols[0],
                    retrieval_cols[1],
                    retrieval_cols[2],
                    result["answer"]
                ]
                writer.writerow(row)
                f.flush()
                
            except Exception as e:
                print(f"Error processing '{question}': {e}")
                writer.writerow([question, category, expected_type, "ERROR", "ERROR", "ERROR", str(e)])

    print(f"\nBatch testing completed! Results saved to: {output_file}")

if __name__ == "__main__":
    batch_test()
