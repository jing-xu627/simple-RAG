import re
from typing import List, Dict
from infra.config import get_config
from indexing.doc_chunk import SimpleChunker, SemanticChunker, ContextualChunker
from indexing.semantic_chunker import IntelligentSemanticChunker

class AdaptiveChunker:
    """自适应分块器 - 根据文档特征选择最佳策略"""
    
    def __init__(self, config):
        self.config = config
        self.simple_chunker = SimpleChunker(config)
        self.semantic_chunker = SemanticChunker(config)
        self.contextual_chunker = ContextualChunker(config)
        self.intelligent_chunker = IntelligentSemanticChunker(config)
    
    def _analyze_document(self, text: str) -> Dict:
        """分析文档特征"""
        return {
            "length": len(text),
            "sentence_count": len(re.split(r'[。！？.!?]', text)),
            "has_structure": bool(re.search(r'[一二三四五六七八九十]、|\d+\.|\n\s*\n', text)),
            "is_technical": bool(re.search(r'[API|算法|模型|系统|技术|数据]', text)),
            "is_formal": bool(re.search(r'[本文|研究|分析|探讨|基于]', text))
        }
    
    def _select_strategy(self, doc_features: Dict) -> str:
        """根据文档特征选择最佳分块策略"""
        length = doc_features["length"]
        sentence_count = doc_features["sentence_count"]
        
        # 短文档 - 使用简单分块
        if length < 1000:
            return "simple"
        
        # 中等文档 - 根据结构选择
        if length < 5000:
            if doc_features["has_structure"]:
                return "semantic"
            else:
                return "contextual"
        
        # 长文档 - 根据质量要求选择
        if length < 20000:
            if doc_features["is_technical"] or doc_features["is_formal"]:
                return "intelligent_semantic"
            else:
                return "contextual"
        
        # 超长文档 - 优先效率
        return "simple"
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """自适应分块主方法"""
        # 分析文档
        features = self._analyze_document(text)
        strategy = self._select_strategy(features)
        
        print(f"[AdaptiveChunker] 文档分析: 长度={features['length']}, 句子={features['sentence_count']}")
        print(f"[AdaptiveChunker] 选择策略: {strategy}")
        
        # 选择对应分块器
        if strategy == "simple":
            chunks = self.simple_chunker.chunk_text(text, doc_id)
        elif strategy == "semantic":
            chunks = self.semantic_chunker.chunk_text(text, doc_id)
        elif strategy == "contextual":
            chunks = self.contextual_chunker.chunk_text(text, doc_id)
        else:  # intelligent_semantic
            chunks = self.intelligent_chunker.chunk_text(text, doc_id)
        
        # 添加策略信息到元数据
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk["metadata"]["chunk_strategy"] = strategy
            else:
                chunk.metadata["chunk_strategy"] = strategy
        
        return chunks
