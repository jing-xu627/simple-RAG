import time
import re
from typing import List, Dict, Optional, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from indexing.embedding import EmbeddingModel
from infra.config import get_config

class IntelligentSemanticChunker:
    """智能语义分块器"""
    def __init__(self, config):
        self.config = config
        self.embedding_model = EmbeddingModel(config)
        self.max_chunk_size = 300  # 最大块大小
        self.min_chunk_size = 50   # 最小块大小
        self.overlap_tokens = 75    # 重叠token数
        self.similarity_threshold = 0.7  # 相似度阈值
    
    def _split_sentences(self, text: str) -> List[str]:
        """智能句子分割"""
        # 清理文本
        text = text.strip()
        if not text:
            return []
        
        # 句子分割模式（中英文混合）
        sentence_patterns = [
            r'[。！？.!?]\s*',  # 中文句号、问号、感叹号 + 英文标点
            r'[;；]\s*',      # 分号
            r'[:：]\s*',      # 冒号
            r'\n\s*\n',       # 段落分隔
            r'\n',            # 换行
        ]
        
        sentences = [text]
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        # 清理和过滤
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # 过滤太短的句子
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _generate_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """并行生成句子向量"""
        print(f"[SemanticChunker] 并行生成 {len(sentences)} 个句子的向量...")
        
        # 准备批次
        batch_size = 10
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        
        def process_batch(batch: List[str]) -> List[List[float]]:
            """处理单个批次的向量生成"""
            try:
                return self.embedding_model.embed(batch)
            except Exception as e:
                print(f"[SemanticChunker] 批次处理失败: {str(e)[:50]}...")
                # Fallback: 逐个处理
                embeddings = []
                for sentence in batch:
                    try:
                        embedding = self.embedding_model.embed_query(sentence)
                        embeddings.append(embedding)
                    except Exception as e2:
                        print(f"[SemanticChunker] 单句处理失败: {str(e2)[:30]}...")
                        embeddings.append([0.0] * 1024)  # 零向量fallback
                return embeddings
        
        # 并行处理所有批次
        all_embeddings = []
        max_workers = min(4, len(batches))  # 最多4个并发
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
            
            # 收集结果
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    all_embeddings.extend(batch_embeddings)
                    print(f"[SemanticChunker] 批次完成: {len(batch)} 个句子")
                except Exception as e:
                    print(f"[SemanticChunker] 批次异常: {str(e)[:50]}...")
                    # 使用零向量作为fallback
                    zero_vector = [0.0] * 1024
                    all_embeddings.extend([zero_vector] * len(batch))
        
        return np.array(all_embeddings)
    
    def _cluster_sentences(self, sentences: List[str], embeddings: np.ndarray) -> List[List[int]]:
        """基于相似度聚类句子"""
        print(f"[SemanticChunker] 开始聚类 {len(sentences)} 个句子...")
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 使用层次聚类思想
        clusters = []
        used = set()
        
        for i, sentence in enumerate(sentences):
            if i in used:
                continue
            
            # 找到相似的句子
            cluster = [i]
            for j in range(i + 1, len(sentences)):
                if j not in used and similarity_matrix[i][j] > self.similarity_threshold:
                    cluster.append(j)
                    used.add(j)
            
            used.add(i)
            clusters.append(cluster)
        
        print(f"[SemanticChunker] 聚类结果: {len(clusters)} 个簇")
        return clusters
    
    def _merge_clusters_to_chunks(self, sentences: List[str], clusters: List[List[int]]) -> List[str]:
        """将聚类合并成块"""
        chunks = []
        
        for cluster in clusters:
            cluster_sentences = [sentences[i] for i in cluster]
            cluster_text = " ".join(cluster_sentences)
            
            # 检查块大小
            if len(cluster_text) > self.max_chunk_size:
                # 如果太长，进一步分割
                chunks.extend(self._split_long_chunk(cluster_text))
            elif len(cluster_text) >= self.min_chunk_size:
                chunks.append(cluster_text)
        
        return chunks
    
    def _split_long_chunk(self, text: str) -> List[str]:
        """分割过长的块"""
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # 寻找合适的分割点
            end_pos = min(current_pos + self.max_chunk_size, len(text))
            
            # 尝试在句子边界分割
            if end_pos < len(text):
                # 在当前范围内寻找最后一个句号
                for i in range(end_pos - 1, current_pos, -1):
                    if text[i] in '。！？.!?；:':
                        end_pos = i + 1
                        break
            
            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算重叠
            current_pos = end_pos - len(chunk) // 4  # 约25%重叠
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """为块添加重叠"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # 计算重叠长度（基于字符数估算token数）
            overlap_chars = min(len(prev_chunk) // 6, 100)  # 约50-100 tokens
            
            # 添加重叠
            if len(prev_chunk) > overlap_chars:
                overlap_text = prev_chunk[-overlap_chars:]
                overlapped_chunk = overlap_text + current_chunk
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """智能语义分块主方法"""
        print(f"[SemanticChunker] 开始智能语义分块，文本长度: {len(text)}")
        
        # 1. 句子分割
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [{"id": f"{doc_id}_0", "doc_id": doc_id, "content": text, "metadata": {"chunk_type": "semantic_single"}}]
        
        # 2. 生成句子向量
        embeddings = self._generate_sentence_embeddings(sentences)
        
        # 3. 语义聚类
        clusters = self._cluster_sentences(sentences, embeddings)
        
        # 4. 合并成块
        chunks = self._merge_clusters_to_chunks(sentences, clusters)
        
        # 5. 添加重叠
        chunks = self._add_overlap(chunks)
        
        # 6. 生成结果
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                "id": f"{doc_id}_semantic_{i}",
                "doc_id": doc_id,  # 添加缺失的doc_id字段
                "content": chunk,
                "metadata": {
                    "chunk_type": "intelligent_semantic",
                    "chunk_index": i,
                    "char_count": len(chunk),
                    "estimated_tokens": len(chunk) // 2  # 粗略估算
                }
            })
        
        print(f"[SemanticChunker] 智能分块完成: {len(result)} 个块")
        return result
