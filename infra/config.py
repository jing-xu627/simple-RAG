import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Ollama配置
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # 默认本地地址
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")  # 默认推荐模型
    
    # DashVector配置
    dashvector_api_key: str = os.getenv("DASHVECTOR_API_KEY", "")
    dashvector_endpoint: str = os.getenv("DASHVECTOR_ENDPOINT", "")
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_collection")
    
    # 分块策略配置
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "contextual")  # simple, semantic, contextual, intelligent_semantic
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "800"))
    min_chunk_size: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "2"))
    
    # 智能语义分块配置
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    overlap_tokens: int = int(os.getenv("OVERLAP_TOKENS", "75"))
    
    # JSON分块配置
    json_max_tokens: int = int(os.getenv("JSON_MAX_TOKENS", "400"))
    json_min_tokens: int = int(os.getenv("JSON_MIN_TOKENS", "50"))
    json_overlap_tokens: int = int(os.getenv("JSON_OVERLAP_TOKENS", "50"))
    
    def validate(self):
        assert self.openai_api_key or self.ollama_base_url, "需要配置OPENAI_API_KEY或OLLAMA_BASE_URL"
        assert self.dashvector_api_key, "DASHVECTOR_API_KEY is required"
        assert self.dashvector_endpoint, "DASHVECTOR_ENDPOINT is required"
        assert self.chunking_strategy in ["simple", "semantic", "contextual", "intelligent_semantic"], "CHUNKING_STRATEGY must be one of: simple, semantic, contextual, intelligent_semantic"
        return self

# 全局配置实例，每次调用时重新加载环境变量
def get_config():
    load_dotenv()  # 重新加载环境变量
    return Config().validate()
