import os
import json
import argparse
from bs4 import BeautifulSoup
from typing import Dict, List, Optional

def extract_laptop_specs(html_content: str) -> Dict:
    """
    Parses the HTML content to extract laptop specifications.
    
    Args:
        html_content (str): The HTML string to parse.
        
    Returns:
        Dict: A dictionary containing the version and list of products with specs.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. 擷取機型名稱
    # 根據 HTML 結構，機型名稱在 gbt-main-checkbox compareCheckBox 中
    model_names = []
    compare_inputs = soup.find_all('input', class_='compareCheckBox')
    for inp in compare_inputs:
        name = inp.get('product-name')
        if name and name not in model_names:
            model_names.append(name)
    
    # 2. 擷取規格標題 (Labels)
    category_labels = []
    title_divs = soup.find_all('div', class_='multiple-title')
    for div in title_divs:
        label = div.get_text(strip=True)
        if label:
            category_labels.append(label)
            
    # 3. 擷取每個產品的具體規格
    products_data = []
    slides = soup.find_all('div', class_='swiper-slide')
    
    # 只取前幾個 slide，數量應與 model_names 一致
    for i, slide in enumerate(slides[:len(model_names)]):
        product_info = {
            "product_name": model_names[i],
            "specs": []
        }
        
        # 尋找該 slide 內的所有規格列
        spec_items = slide.find_all('div', class_='spec-item-list')
        for item in spec_items:
            row_idx_str = item.get('data-spec-row')
            if row_idx_str is not None:
                row_idx = int(row_idx_str)
                # 取得規格值內容
                spec_value = item.get_text(separator="\n", strip=True)
                
                # 清除可能帶有的行號 (例如 "219:", "220:")
                import re
                spec_value = re.sub(r'^\d+:\s*', '', spec_value, flags=re.MULTILINE)
                
                # 對應標題
                if row_idx < len(category_labels):
                    category = category_labels[row_idx]
                    product_info["specs"].append({
                        "category": category,
                        "value": spec_value
                    })
        
        products_data.append(product_info)

    return {
        "version": "1.0-raw",
        "products": products_data
    }

def main():
    parser = argparse.ArgumentParser(description="Parse Laptop Specs HTML to JSON")
    parser.add_argument("--input", "-i", default="AORUS MASTER 16 AM6H.html", help="Path to the input HTML file")
    parser.add_argument("--output", "-o", default="data/raw_specs.json", help="Path to the output JSON file")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if os.path.exists(input_file):
        print(f"Reading HTML file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            data = extract_laptop_specs(content)
            
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, indent=4, ensure_ascii=False)
            
        print(f"Successfully extracted {len(data['products'])} products.")
        print(f"Saved to: {output_file}")
    else:
        print(f"Error: Input file not found: {input_file}")

if __name__ == "__main__":
    main()
