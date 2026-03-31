import asyncio
import json
import re
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
from indexing.embedding import EmbeddingModel
from infra.config import get_config
from indexing.semantic_chunker import IntelligentSemanticChunker
from indexing.json_chunker import JsonChunker

@dataclass
class DocumentChunk:
    id: str
    content: str
    doc_id: str
    metadata: Dict
    page_num: Optional[int] = None
    position: Optional[int] = None

class SimpleChunker:
    def __init__(self, config):
        self.config = config
        self.chunk_size = 1000  # 字符数
        self.chunk_overlap = 200  # 重叠字符数
    
    def chunk_text(self, text: str, doc_id: str, 
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> List[DocumentChunk]:
        """简单的文本分块"""
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        chunk_index = 0
        if len(text) <= chunk_size:
            return [DocumentChunk(
                id=f"{doc_id}_{chunk_index}",
                content=text,
                doc_id=doc_id,
                metadata={"chunk_index": chunk_index}
            )]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 如果不是最后一块，尝试在句子边界分割
            if end < len(text):
                # 寻找最近的句号、问号、感叹号
                sentence_end = max(
                    text.rfind('。', start, end),
                    text.rfind('？', start, end),
                    text.rfind('！', start, end),
                    text.rfind('.', start, end),
                    text.rfind('?', start, end),
                    text.rfind('!', start, end),
                    end - 1  # 如果找不到句子结束符，使用原始边界
                )
                
                # 如果找到合适的句子边界且不会让块太小
                if sentence_end > start + chunk_size * 0.3:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(DocumentChunk(
                    id=f"{doc_id}_{chunk_index}",
                    content=chunk_text,
                    doc_id=doc_id,
                    metadata={
                        "chunk_index": chunk_index,
                        "start_pos": start,
                        "end_pos": end
                    }
                ))
                chunk_index += 1
            
            start = end - chunk_overlap if end - chunk_overlap > start else end
        
        return chunks

class SemanticChunker:
    """语义感知分块器"""
    def __init__(self, config):
        self.config = config
        self.max_chunk_size = 800  # 语义分块通常小一些
        self.min_chunk_size = 100
        self.sentence_overlap = 2  # 句子重叠数量
    
    def chunk_text(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """基于语义的文本分块"""
        # 1. 按句子分割
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [DocumentChunk(
                id=f"{doc_id}_0",
                content=text,
                doc_id=doc_id,
                metadata={"chunk_type": "semantic", "sentence_count": len(sentences)}
            )]
        
        # 2. 语义分块
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查是否需要开始新块
            if (current_length + len(sentence) > self.max_chunk_size and 
                current_length >= self.min_chunk_size):
                
                # 创建当前块
                chunk_content = " ".join(current_chunk_sentences)
                chunks.append(DocumentChunk(
                    id=f"{doc_id}_{chunk_index}",
                    content=chunk_content,
                    doc_id=doc_id,
                    metadata={
                        "chunk_type": "semantic",
                        "sentence_count": len(current_chunk_sentences),
                        "char_count": len(chunk_content),
                        "start_sentence": i - len(current_chunk_sentences),
                        "end_sentence": i - 1
                    }
                ))
                
                # 开始新块，保留重叠句子
                current_chunk_sentences = current_chunk_sentences[-self.sentence_overlap:]
                current_length = sum(len(s) for s in current_chunk_sentences)
                chunk_index += 1
            
            current_chunk_sentences.append(sentence)
            current_length += len(sentence)
        
        # 处理最后一块
        if current_chunk_sentences:
            chunk_content = " ".join(current_chunk_sentences)
            chunks.append(DocumentChunk(
                id=f"{doc_id}_{chunk_index}",
                content=chunk_content,
                doc_id=doc_id,
                metadata={
                    "chunk_type": "semantic",
                    "sentence_count": len(current_chunk_sentences),
                    "char_count": len(chunk_content),
                    "start_sentence": len(sentences) - len(current_chunk_sentences),
                    "end_sentence": len(sentences) - 1
                }
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """智能句子分割"""
        # 中英文句子分割
        sentence_patterns = [
            r'[。！？.!?]\s*',  # 中文句号、问号、感叹号 + 英文标点
            r'[;；]\s*',      # 分号
            r'\n\s*\n',       # 段落分隔
        ]
        
        # 先按段落分割
        paragraphs = re.split(r'\n\s*\n', text.strip())
        sentences = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 在段落内按句子分割
            para_sentences = []
            current = ""
            i = 0
            
            while i < len(paragraph):
                current += paragraph[i]
                
                # 检查是否是句子结束符
                if paragraph[i] in '。！？.!?；;':
                    # 检查下一个字符是否是引号或括号
                    if i + 1 < len(paragraph):
                        next_chars = paragraph[i+1:i+3]  # 获取后面2个字符
                        if not any(c in next_chars for c in '"」）'):
                            para_sentences.append(current.strip())
                            current = ""
                
                i += 1
            
            # 添加剩余内容
            if current.strip():
                para_sentences.append(current.strip())
            
            sentences.extend(para_sentences)
        
        return [s for s in sentences if s]

class ContextualChunker:
    """上下文感知分块器"""
    def __init__(self, config):
        self.config = config
        self.semantic_chunker = SemanticChunker(config)
        # 从配置读取上下文窗口大小，默认为2
        self.context_window = getattr(config, 'context_window', 2)
    
    def chunk_text(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """生成带上下文的分块"""
        # 1. 先进行语义分块
        semantic_chunks = self.semantic_chunker.chunk_text(text, doc_id)
        
        if len(semantic_chunks) <= 1:
            return semantic_chunks
        
        # 2. 为每个块添加上下文
        contextual_chunks = []
        
        for i, chunk in enumerate(semantic_chunks):
            # 获取上下文块
            start_idx = max(0, i - self.context_window)
            end_idx = min(len(semantic_chunks), i + self.context_window + 1)
            
            context_chunks = semantic_chunks[start_idx:end_idx]
            context_content = " ".join([c.content for c in context_chunks])
            
            # 创建上下文增强的块
            contextual_chunk = DocumentChunk(
                id=f"{doc_id}_ctx_{i}",
                content=chunk.content,  # 保持原始内容
                doc_id=doc_id,
                metadata={
                    "chunk_type": "contextual",
                    "original_chunk_id": chunk.id,
                    "context_window": f"{start_idx}-{end_idx}",
                    "context_content": context_content,
                    "position_in_context": i - start_idx,
                    "total_context_chunks": len(context_chunks),
                    **chunk.metadata  # 保留原始元数据
                }
            )
            contextual_chunks.append(contextual_chunk)
        
        return contextual_chunks

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.embedding_model = EmbeddingModel(config)
        
        # 初始化所有分块器
        self.simple_chunker = SimpleChunker(config)
        self.semantic_chunker = SemanticChunker(config)
        self.contextual_chunker = ContextualChunker(config)
        self.intelligent_chunker = IntelligentSemanticChunker(config)
        self.json_chunker = JsonChunker(config)
        
        # 根据配置选择默认分块策略
        self.chunking_strategy = getattr(config, 'chunking_strategy', 'contextual')
    
    def _detect_content_type(self, text: str, file_name: str = "") -> str:
        """检测内容类型"""
        # 检查文件扩展名
        if file_name.lower().endswith('.json'):
            return 'json'
        
        # 检查内容是否为JSON
        try:
            json.loads(text.strip())
            return 'json'
        except:
            pass
        
        # 检查是否为结构化数据
        if re.search(r'^\s*[\[\{]', text.strip()):
            try:
                json.loads(text.strip())
                return 'json'
            except:
                pass
        
        return 'text'
    
    def _select_chunker(self, content_type: str):
        """根据内容类型选择分块器"""
        if content_type == 'json':
            return self.json_chunker
        
        # 根据配置选择文本分块器
        if self.chunking_strategy == "intelligent_semantic":
            return self.intelligent_chunker
        elif self.chunking_strategy == "semantic":
            return self.semantic_chunker
        elif self.chunking_strategy == "contextual":
            return self.contextual_chunker
        else:
            return self.simple_chunker
    
    async def process_document(self, text: str, doc_id: str, file_name: str = "",
                              progress_callback: Optional[Callable] = None) -> List[Dict]:
        """处理文档：分块 + embedding"""
        
        # 1. 检测内容类型
        content_type = self._detect_content_type(text, file_name)
        print(f"[DocumentProcessor] 检测到内容类型: {content_type}")
        
        # 2. 选择分块器
        chunker = self._select_chunker(content_type)
        
        # 3. 文档分块
        if progress_callback:
            progress_callback(0.1, "开始文档分块...")
        
        # 分块返回字典格式
        chunk_dicts = chunker.chunk_text(text, doc_id)
        
        if progress_callback:
            progress_callback(0.3, f"文档分块完成，共{len(chunk_dicts)}个块")
        
        # 4. 为分块结果生成embedding
        if progress_callback:
            progress_callback(0.4, "开始生成向量...")
        
        # 提取文本内容用于向量化
        chunk_texts = [chunk["content"] for chunk in chunk_dicts]
        chunk_embeddings = await self._generate_embeddings_for_chunks(chunk_texts, progress_callback)
        
        if progress_callback:
            progress_callback(0.8, "向量生成完成")
        
        # 5. 组装结果
        processed_chunks = []
        for i, chunk_dict in enumerate(chunk_dicts):
            chunk_dict["embedding"] = chunk_embeddings[i]
            # 添加内容类型信息
            chunk_dict["metadata"]["content_type"] = content_type
            processed_chunks.append(chunk_dict)
        
        if progress_callback:
            progress_callback(1.0, f"文档处理完成，共处理{len(processed_chunks)}个块")
        
        return processed_chunks
    
    async def _generate_embeddings_for_chunks(self, chunk_texts: List[str], 
                                             progress_callback: Optional[Callable] = None,
                                             batch_size: int = 5) -> List[List[float]]:
        """为chunk文本生成embedding"""
        all_embeddings = []
        total_batches = (len(chunk_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[i:i + batch_size]
            
            # 限制文本长度避免API限制
            safe_texts = []
            for text in batch_texts:
                if len(text) > 2000:  # 智能分块通常更短
                    text = text[:2000]
                safe_texts.append(text)
            
            # 批量生成embedding
            try:
                batch_embeddings = self.embedding_model.embed(safe_texts)
                all_embeddings.extend(batch_embeddings)
                print(f"[DocumentProcessor] Batch {i//batch_size + 1}: 成功处理 {len(batch_texts)} 个块")
            except Exception as e:
                print(f"[DocumentProcessor] Batch {i//batch_size + 1} 失败: {str(e)[:100]}...")
                # Fallback策略
                fallback_texts = [text[:800] for text in batch_texts]
                batch_embeddings = self.embedding_model.embed(fallback_texts)
                all_embeddings.extend(batch_embeddings)
                print(f"[DocumentProcessor] Batch {i//batch_size + 1} Fallback: 成功处理 {len(batch_texts)} 个块")
            
            if progress_callback:
                current_batch = i // batch_size + 1
                progress_bar = 0.4 + 0.4 * current_batch / total_batches
                progress_callback(progress_bar, f"生成向量进度: {current_batch}/{total_batches} 批次")
            
            # 避免API限制，添加延迟
            await asyncio.sleep(0.2)
        
        return all_embeddings
    
    async def process_multiple_documents(self, documents: List[Dict[str, str]],
                                       progress_callback: Optional[Callable] = None) -> List[Dict]:
        """处理多个文档"""
        all_chunks = []
        total_docs = len(documents)
        
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{i}")
            text = doc["text"]
            
            if progress_callback:
                doc_progress = i / total_docs
                progress_callback(doc_progress, f"处理文档 {i+1}/{total_docs}: {doc_id}")
            
            chunks = await self.process_document(text, doc_id, progress_callback)
            all_chunks.extend(chunks)
        
        if progress_callback:
            progress_callback(1.0, f"所有文档处理完成，共生成{len(all_chunks)}个块")
        
        return all_chunks

# 使用示例
async def main():
    config = get_config()
    processor = DocumentProcessor(config)
    
    # 示例文档
    documents = [
        {
            "id": "doc1",
            "text": """Python是一种高级编程语言，由Guido van Rossum于1991年创建。它以简洁、易读的语法著称。
            
Python的设计哲学强调代码的可读性和简洁性。它支持多种编程范式，包括面向对象、函数式和过程式编程。

Python拥有丰富的标准库和第三方库生态系统，使其成为数据科学、机器学习、Web开发等领域的热门选择。"""
        },
        {
            "id": "doc2", 
            "text": """机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需明确编程。

机器学习算法通过分析训练数据来识别模式，并基于这些模式做出预测或决策。常见的机器学习类型包括监督学习、无监督学习和强化学习。

深度学习是机器学习的一个子领域，使用神经网络来模拟人脑的学习过程。"""
        }
    ]
    
    def progress_callback(progress: float, message: str):
        print(f"[{progress:.1%}] {message}")
    
    # 处理文档
    chunks = await processor.process_multiple_documents(documents, progress_callback)
    
    print(f"\n处理结果:")
    for chunk in chunks[:3]:  # 显示前3个块
        print(f"块ID: {chunk['id']}")
        print(f"文档ID: {chunk['doc_id']}")
        print(f"内容: {chunk['content'][:100]}...")
        print(f"向量维度: {len(chunk['embedding'])}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
