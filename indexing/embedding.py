try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    # 如果没有安装 langchain_ollama，使用旧版本
    from langchain_community.embeddings import OllamaEmbeddings
    import warnings
    warnings.warn("建议安装 langchain-ollama: pip install langchain-ollama", DeprecationWarning)
from typing import List
from infra.config import Config

class EmbeddingModel:
    def __init__(self, config):
        self.embed_model = OllamaEmbeddings(
            model=config.ollama_embedding_model,
            base_url=config.ollama_base_url,
        )
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """文本向量化"""
        if isinstance(texts, str):
            texts = [texts]
        
        return self.embed_model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """单个查询文本向量化"""
        return self.embed_model.embed_query(text)

    def get_embedding(self, text: str) -> List[float]:
        """兼容DashVectorDb接口"""
        return self.embed_query(text)
