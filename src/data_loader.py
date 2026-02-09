"""
資料載入模組

此模組提供產品規格資料的載入和轉換功能。
將 JSON 格式的產品規格資料轉換為可供 RAG 系統使用的 Chunk 物件。
"""
import json
from typing import List, Dict, Optional
import os
try:
    from src.models import Chunk
except ImportError:
    from models import Chunk


class ConvertSpec:
    """
    規格轉換器類別
    
    功用：
        從 JSON 檔案載入產品規格資料，並將其轉換為 Chunk 物件，
        用於後續的向量索引和檢索。
    
    類別屬性：
        CANONICAL_TO_KEYS (dict): 規格類別到相關關鍵字的映射，用於增強搜尋效果
    
    屬性：
        filepath (str): JSON 資料檔案路徑
    """
    
    # Category to related keywords mapping (for keyword search boost)
    CANONICAL_TO_KEYS = {
        "gpu": ["GPU", "Graphics", "gaming", "顯卡", "顯示卡", "遊戲"],
        "cpu": ["CPU", "Processor", "中央處理器", "處理器"],
        "ram": ["RAM", "Memory", "記憶體"],
        "ssd": ["SSD", "Storage", "儲存裝置", "硬碟"],
        "display": ["Screen", "Display", "HDMI", "DP", "顯示器", "螢幕"],
        "battery": ["Battery", "電池"],
        "ports": ["Ports", "Connectivity", "連接埠", "I/O", "USB","Thunderbolt", "HDMI"],
        "wireless": ["Wireless", "Networking", "通訊", "WiFi"],
        "audio": ["Audio", "Speakers", "音效", "喇叭"],
        "dimensions": ["Dimensions", "Physical", "尺寸"],
        "weight": ["Weight", "Physical", "重量"],
    }
    
    def __init__(self, filepath: str = "data_processing/knowledge_base/specs_integrated.json"):
        """
        初始化規格轉換器
        
        功用：
            建立轉換器實例並設定資料檔案路徑。
        
        輸入：
            filepath (str): JSON 資料檔案路徑（預設為 "data_processing/knowledge_base/specs_integrated.json"）
        
        輸出：
            無（初始化實例）
        """
        self.filepath = filepath

    def load_data(self) -> Dict:
        """
        載入 JSON 資料
        
        功用：
            從指定的 JSON 檔案讀取完整的資料結構。
        
        輸入：
            無
        
        輸出：
            Dict: 包含產品規格資料的字典
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_related_words(self, category: str, key: str) -> str:
        """
        取得相關關鍵字
        
        功用：
            根據規格類別或鍵值取得相關的搜尋關鍵字，用於增強搜尋效果。
        
        輸入：
            category (str): 規格類別（如 "GPU"、"CPU"）
            key (str): 規格鍵值（如 "Graphics"、"Processor"）
        
        輸出：
            str: 包含相關關鍵字的字串，格式為 "Related words: 關鍵字1、關鍵字2"，
                 若無匹配則返回空字串
        """
        search_terms = [category.lower(), key.lower()]
        
        for canonical, keywords in self.CANONICAL_TO_KEYS.items():
            # Check if category or key matches any keyword in the mapping
            for kw in keywords:
                if kw.lower() in search_terms or any(kw.lower() in term for term in search_terms):
                    return "Related words: " + "、".join(keywords)
        
        return ""

    def chunk_data(self) -> List[Chunk]:
        """
        將規格資料轉換為區塊列表
        
        功用：
            將 JSON 格式的產品規格資料轉換為 Chunk 物件列表，
            每個規格項目會生成一個獨立的區塊，同時為每個產品生成一個完整規格區塊。
        
        輸入：
            無
        
        輸出：
            List[Chunk]: 包含所有產品規格的區塊列表，每個區塊包含：
                - content: 格式化的規格文本
                - metadata: 包含 product_id、product_name、category、key 的元資料
        """
        data = self.load_data()
        chunks = []
        
        for product in data["products"]:
            product_id = product.get("product_id", "unknown")
            product_name = product.get("product_name", "Unknown Product")
            
            # Create individual spec chunks
            for spec in product.get("specs", []):
                category = spec.get('category', '')
                key = spec.get('key', '')
                
                content_parts = [
                    f"Product: {product_name}",
                    f"Category: {category}",
                    f"Specification: {key}",
                    f"Value: {spec.get('value', '')}",
                ]
                
                # Add related words if available
                related_words = self._get_related_words(category, key)
                if related_words:
                    content_parts.append(related_words)
                
                text = "\n".join(content_parts)
                
                chunks.append(Chunk(
                    content=text,
                    metadata={
                        "product_id": product_id,
                        "product_name": product_name,
                        "category": category,
                        "key": key,
                    }
                ))
            
            # Create "Full Machine Specs" chunk
            full_specs_content = []
            for spec in product.get("specs", []):
                cat = spec.get("category", "")
                key = spec.get("key", "")
                val = spec.get("value", "").replace("\n", " ")
                full_specs_content.append(f"- {cat} - {key}: {val}")
            
            full_specs_text_body = "\n".join(full_specs_content)
            
            full_specs_parts = [
                f"Product: {product_name}",
                "Category: Overall Specification",
                "Specification: Full Specs",
                "Value:",
                full_specs_text_body,
                "Related terms: specifications, full specs, configuration, computer, laptop, overall, 規格, 整機規格, 配置, 筆電, 電腦"
            ]
            
            chunks.append(Chunk(
                content="\n".join(full_specs_parts),
                metadata={
                    "product_id": product_id,
                    "product_name": product_name,
                    "category": "Overall Specification",
                    "key": "Full Specs"
                }
            ))
        
        return chunks
    
    def get_products(self) -> List[Dict]:
        """
        取得產品列表
        
        功用：
            返回所有產品的完整資訊列表。
        
        輸入：
            無
        
        輸出：
            List[Dict]: 產品字典列表，每個字典包含產品的完整資訊
        """
        data = self.load_data()
        return data.get("products", [])
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """
        根據 ID 取得產品
        
        功用：
            根據產品 ID 查找並返回對應的產品資訊。
        
        輸入：
            product_id (str): 產品 ID（如 "aorus-master-16-bxh"）
        
        輸出：
            Optional[Dict]: 產品字典（若找到）或 None（若未找到）
        """
        for product in self.get_products():
            if product.get("product_id") == product_id:
                return product
        return None

    def get_product_specs(self, product_id: str) -> Dict[str, str]:
        """
        取得產品規格字典
        
        功用：
            返回指定產品的扁平化規格字典，方便進行確定性查找。
        
        輸入：
            product_id (str): 產品 ID（如 "aorus-master-16-bxh"）
        
        輸出：
            Dict[str, str]: 規格字典，格式為 {規格鍵: 規格值}，
                同時包含純鍵值和帶類別前綴的鍵值
        """
        product = self.get_product_by_id(product_id)
        if not product:
            return {}
        
        specs_dict = {}
        for spec in product.get("specs", []):
            key = spec.get("key", "").lower()
            val = spec.get("value", "")
            specs_dict[key] = val
            # Also store with category prefix
            cat = spec.get("category", "").lower()
            specs_dict[f"{cat} {key}"] = val
            
        return specs_dict


if __name__ == "__main__":
    loader = ConvertSpec("data_processing/knowledge_base/specs_integrated.json")
    chunks = loader.chunk_data()
    print(f"Total chunks: {len(chunks)}")
    

    c = chunks[:5]
    for i, chunk in enumerate(c):
        print(f"\n--- Chunk {i+1} Example ---")
        print(f"Content:\n{chunk.content}")
        print(f"Metadata: {chunk.metadata}")
