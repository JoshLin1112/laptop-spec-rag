"""
查詢分析模組

此模組提供查詢分析功能，包括查詢過濾器（QueryFilter）和產品名稱提取器（ProductNameExtractor）。
用於識別使用者查詢的意圖和提取產品資訊。
"""
from typing import List, Dict, Optional, Set

# Support both direct execution and module import
try:
    from src.models import Chunk
except ImportError:
    from models import Chunk


class QueryFilter:
    """
    查詢過濾器類別
    
    功用：
        分析使用者查詢，識別查詢類型（一般規格、產品列表、產品比較等）。
    
    類別屬性：
        CANONICAL_TO_KEYS (dict): 規格類別到關鍵字的映射
        CORE_CATEGORIES (list): 核心規格類別列表（CPU、GPU、RAM、顯示器）
        GENERAL_SPEC_KEYWORDS (list): 一般規格查詢關鍵字
        PRODUCT_LIST_KEYWORDS (list): 產品列表查詢關鍵字
        COMPARISON_KEYWORDS (list): 產品比較查詢關鍵字
    """
    
    # === Category Mappings ===
    CANONICAL_TO_KEYS = {
        "gpu": ["GPU", "Graphics", "gaming", "顯卡", "顯示卡", "遊戲"],
        "cpu": ["CPU", "Processor", "中央處理器", "處理器"],
        "ram": ["RAM", "Memory", "記憶體"],
        "ssd": ["SSD", "Storage", "儲存裝置", "硬碟"],
        "display": ["Screen", "Display","HDMI", "DP" ,"顯示器", "螢幕"],
        "battery": ["Battery", "電池"],
        "ports": ["Ports", "Connectivity", "連接埠", "I/O", "usb"],
        "wireless": ["Wireless", "Networking", "通訊", "WiFi"],
        "audio": ["Audio", "Speakers", "音效", "喇叭"],
        "dimensions": ["Dimensions", "Physical", "尺寸"],
        "weight": ["Weight", "Physical", "重量"],
    }
    
    CORE_CATEGORIES = ["cpu", "gpu", "ram", "display"]
    
    # === Intent Keywords ===
    GENERAL_SPEC_KEYWORDS = [
        "規格", "specs", "specifications", "spec", "配置", 
        "筆電", "laptop", "電腦", "computer", "這台", "這款"
    ]
    
    PRODUCT_LIST_KEYWORDS = [
        "有哪些產品", "甚麼產品", "什麼產品", "what products", "list products", 
        "available models", "所有產品", "all products", "有哪幾台", "有哪幾款"
    ]
    
    COMPARISON_KEYWORDS = [
        "比較", "差異", "差別", "不同", "compare", "comparison", 
        "difference", "vs", "versus", "產品"
    ]
    
    def __init__(self):
        """
        初始化查詢過濾器
        
        功用：
            建立查詢過濾器實例。
        
        輸入：
            無
        
        輸出：
            無（初始化實例）
        """
        pass
    
    def is_general_spec_query(self, query: str) -> bool:
        """
        檢查是否為一般規格查詢
        
        功用：
            判斷使用者查詢是否在詢問一般性的產品規格。
        
        輸入：
            query (str): 使用者查詢文本
        
        輸出：
            bool: 如果是一般規格查詢則返回 True，否則返回 False
        """
        query_lower = query.lower()
        for keyword in self.GENERAL_SPEC_KEYWORDS:
            if keyword.lower() in query_lower:
                return True
        return False
        
    def is_product_list_query(self, query: str) -> bool:
        """
        檢查是否為產品列表查詢
        
        功用：
            判斷使用者查詢是否在詢問可用的產品列表。
        
        輸入：
            query (str): 使用者查詢文本
        
        輸出：
            bool: 如果是產品列表查詢則返回 True，否則返回 False
        """
        query_lower = query.lower()
        for keyword in self.PRODUCT_LIST_KEYWORDS:
            if keyword.lower() in query_lower:
                return True
        return False
    
    def is_comparison_query(self, query: str) -> bool:
        """
        檢查是否為產品比較查詢
        
        功用：
            判斷使用者查詢是否在詢問產品之間的比較。
        
        輸入：
            query (str): 使用者查詢文本
        
        輸出：
            bool: 如果是產品比較查詢則返回 True，否則返回 False
        """
        query_lower = query.lower()
        for keyword in self.COMPARISON_KEYWORDS:
            if keyword.lower() in query_lower:
                return True
        return False


class ProductNameExtractor:
    """
    產品名稱提取器類別
    
    功用：
        從使用者查詢中提取產品型號後綴（如 BXH、BYH、BZH）。
    
    屬性：
        known_models (list): 已知產品型號列表，格式為 (product_id, product_name) 元組
        fallback_suffixes (list): 備用型號後綴列表（當無動態列表時使用）
    """
    
    def __init__(self, product_list: List[Dict] = None):
        """
        初始化產品名稱提取器
        
        功用：
            建立提取器實例，載入已知產品列表。
        
        輸入：
            product_list (List[Dict]): 產品字典列表（可選），
                每個字典應包含 "product_id" 和 "product_name" 鍵
        
        輸出：
            無（初始化實例）
        """
        self.known_models = []
        if product_list:
            for p in product_list:
                self.known_models.append((p.get("product_id"), p.get("product_name")))
        
        # Fallback hardcoded if no list provided (backward compatibility)
        if not self.known_models:
            self.fallback_suffixes = ["BXH", "BYH", "BZH"]

    def extract(self, query: str) -> Optional[str]:
        """
        從查詢中提取產品 ID
        
        功用：
            使用模糊匹配從使用者查詢中提取產品 ID。
        
        輸入：
            query (str): 使用者查詢文本
        
        輸出：
            Optional[str]: 產品 ID（如 "aorus-master-16-bxh"）或 None（若未找到）
        """
        query_upper = query.upper()
        
        # 1. Check against loaded dynamic list
        if self.known_models:
            for pid, pname in self.known_models:
                suffix = pid.split("-")[-1].upper()
                if suffix in query_upper:
                    return pid
        
        # 2. Fallback to hardcoded suffixes
        if hasattr(self, 'fallback_suffixes'):
            for model in self.fallback_suffixes:
                if model in query_upper:
                    return model
        
        return None
