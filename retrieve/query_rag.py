import asyncio
from typing import List, Dict, Optional
from infra.config import get_config
from infra.llm import LLM
from infra.dashvector_db import DashVectorDb, DashVectorConfig
from indexing.embedding import EmbeddingModel
from agno.knowledge.document import Document

class RAGQuery:
    """RAG查询处理器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm = LLM(self.config)
        self.embedding_model = EmbeddingModel(self.config)
        
        # 自动获取embedding维度
        if not hasattr(self.embedding_model, 'dimension'):
            # 通过测试embedding获取维度
            test_embedding = self.embedding_model.embed_query("test")
            auto_dimension = len(test_embedding)
        else:
            auto_dimension = self.embedding_model.dimension
        
        print(f"[RAGQuery] Auto-detected embedding dimension: {auto_dimension}")
        
        # 初始化DashVector数据库
        dashvector_config = DashVectorConfig(
            api_key=self.config.dashvector_api_key,
            endpoint=self.config.dashvector_endpoint,
        )
        
        self.vector_db = DashVectorDb(
            collection=self.config.collection_name,
            embedder=self.embedding_model,
            config=dashvector_config,
            dimension=auto_dimension,  # 使用自动检测的维度
            metric="cosine",
        )
    
    async def query(self, question: str, top_k: int = 5, 
                  system_prompt: Optional[str] = None) -> Dict:
        """完整的RAG查询流程"""
        try:
            # 1. 将查询转换为向量
            query_vector = self.embedding_model.embed_query(question)
            
            # 2. 搜索相关文档
            search_results = await self.vector_db.async_search(
                query=question,
                limit=top_k
            )
            
            print(f"[RAGQuery] Found {len(search_results)} relevant documents")
            
            # 3. 构建上下文
            if search_results:
                context_parts = []
                for i, doc in enumerate(search_results, 1):
                    context_parts.append(f"Document {i}:\n{doc.content[:1000]}")
                context = "\n\n".join(context_parts)
            else:
                context = "No relevant documents found."
            
            print(f"[RAGQuery] Context length: {len(context)} chars")
            
            # 4. 构建Prompt
            default_system_prompt = "你是一个基于检索增强的问答助手。请根据提供的上下文回答问题，如果上下文不足以回答问题，请明确说明。"
            
            prompt = f"""仔细阅读以下上下文，直接回答用户问题。
                    重要规则：
                    1. 基于上下文中的具体内容回答，不可随意捏造
                    2. 不要过度解读，文中没说的就说不知道
                    上下文：
                    {context}
                    用户问题：{question}
                    请给出详细且准确的回答"""
            
            # 5. 生成回答
            answer = self.llm.generate(prompt, system_prompt or default_system_prompt)
            
            return {
                "answer": answer,
                "sources": search_results,
                "context": context,
                "question": question
            }
            
        except Exception as e:
            return {
                "answer": f"查询过程中发生错误: {str(e)}",
                "sources": [],
                "context": "",
                "question": question,
                "error": str(e)
            }

# 便捷函数
async def rag_query(question: str, config: Optional[Config] = None,
                 top_k: int = 5, system_prompt: Optional[str] = None) -> Dict:
    """便捷的RAG查询函数"""
    rag = RAGQuery(config)
    return await rag.query(question, top_k, system_prompt)