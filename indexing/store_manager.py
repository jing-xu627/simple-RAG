import asyncio
from typing import List, Dict, Optional, Callable
from infra.config import get_config
from infra.dashvector_db import DashVectorDb, DashVectorConfig
from indexing.embedding import EmbeddingModel
from agno.knowledge.document import Document

class VectorStoreManager:
    def __init__(self, config=None):
        self.config = config or get_config()
    
    async def store_chunks(self, chunks: List[Dict], 
                          progress_callback: Optional[Callable] = None) -> bool:
        """将处理后的chunks存储到向量数据库"""
        try:
            if progress_callback:
                progress_callback(0.9, "存储到向量数据库...")
            
            # 自动获取embedding维度
            embedding_model = EmbeddingModel(self.config)
            test_embedding = embedding_model.embed_query("test")
            auto_dimension = len(test_embedding)
            
        
            # 设置DashVector配置
            dashvector_config = DashVectorConfig(
                api_key=self.config.dashvector_api_key,
                endpoint=self.config.dashvector_endpoint,
            )
            
            vector_db = DashVectorDb(
                collection=self.config.collection_name,
                embedder=embedding_model,
                config=dashvector_config,
                dimension=auto_dimension,  # 使用自动检测的维度
                metric="cosine",
            )
            
            # 将chunks转换为Document对象
            documents = []
            for i, chunk in enumerate(chunks):
                
                # 确保chunk是字典类型
                if not isinstance(chunk, dict):
                    print(f"[VectorStoreManager] Error: Chunk {i} is not a dict, type: {type(chunk)}")
                    continue
                
                # 检查必需的键
                required_keys = ["id", "content", "embedding", "metadata"]
                for key in required_keys:
                    if key not in chunk:
                        print(f"[VectorStoreManager] Error: Chunk {i} missing key: {key}")
                        continue
                
                doc = Document(
                    id=chunk["id"],
                    content=chunk["content"],
                    embedding=chunk["embedding"],
                    meta_data=chunk["metadata"]
                )
                documents.append(doc)
            
            # 存储到向量库
            await vector_db.async_insert(documents)
            
            if progress_callback:
                progress_callback(1.0, f"成功存储{len(documents)}个文档块")
            
            print(f"[VectorStoreManager] 实际存储: {len(documents)}/{len(chunks)} 个块")
            return True
            
        except Exception as e:
            raise Exception(f"存储到向量数据库失败: {e}")