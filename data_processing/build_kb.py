import json
import os
import argparse

def integrate_and_simplify_specs(input_file: str, output_file: str):
    """
    Integrates raw specs into the final knowledge base format.
    
    Args:
        input_file (str): Path to raw specs JSON.
        output_file (str): Path to output integrated JSON.
    """
    # 1. 讀取原始爬取的資料
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Reading raw specs from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 2. 定義分類映射表 (中 -> 英) 以及對應的 Key
    # 根據 specs.json 的結構進行映射
    mapping = {
        "作業系統": ("Processor", "OS"),
        "中央處理器": ("Processor", "CPU"),
        "顯示晶片": ("Graphics", "GPU"),
        "顯示器": ("Display", "Screen"),
        "記憶體": ("Memory", "RAM"),
        "儲存裝置": ("Storage", "SSD"),
        "鍵盤種類": ("Keyboard", "Keyboard Type"),
        "連接埠": ("Connectivity", "Ports"),
        "音效": ("Audio", "Speakers"),
        "通訊": ("Networking", "Wireless"),
        "視訊鏡頭": ("Webcam", "Webcam"),
        "安全裝置": ("Security", "Security"),
        "電池": ("Battery", "Battery"),
        "變壓器": ("Power", "Adapter"),
        "尺寸": ("Physical", "Dimensions"),
        "重量": ("Physical", "Weight"),
        "顏色": ("Physical", "Color")
    }

    new_products = []
    
    for product in raw_data.get("products", []):
        model_name = product.get("product_name", "")
        
        # 建立英文版 product_id
        product_id = model_name.lower().replace(" ", "-")
        
        new_product = {
            "product_id": product_id,
            "product_name": model_name,
            "specs": []
        }
        
        for spec in product.get("specs", []):
            zh_cat = spec.get("category", "")
            value = spec.get("value", "")
            
            if zh_cat in mapping:
                en_cat, en_key = mapping[zh_cat]
                new_product["specs"].append({
                    "category": en_cat,
                    "key": en_key,
                    "value": value
                })
            else:
                # 沒在映射表裡的就直接用原始名稱
                new_product["specs"].append({
                    "category": zh_cat,
                    "key": zh_cat,
                    "value": value
                })
        
        new_products.append(new_product)

    # 3. 組合最終 JSON
    # 注意：Synonyms 資料已獨立至 data/synonyms.json，不再包含於此
    final_json = {
        "version": "2.1",
        "products": new_products
    }

    # 4. 寫入檔案
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully created knowledge base at: {output_file}")
    print("Format has been simplified to English Keys. Synonyms are stored separately.")

def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Base from Raw Specs")
    parser.add_argument("--input", "-i", default="raw/raw_specs.json", help="Path to raw specs JSON")
    parser.add_argument("--output", "-o", default="knowledge_base/specs_integrated.json", help="Path to output Knowledge Base JSON")
    
    args = parser.parse_args()
    
    integrate_and_simplify_specs(args.input, args.output)

if __name__ == "__main__":
    main()
