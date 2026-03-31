import json
import re
from typing import List, Dict, Any, Optional
from indexing.embedding import EmbeddingModel
from infra.config import get_config

class JsonChunker:
    """JSON智能分块器"""
    
    def __init__(self, config):
        self.config = config
        self.max_tokens = 400  # 最大token数
        self.min_tokens = 50   # 最小token数
        self.overlap_tokens = 50  # 重叠token数
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数量 (粗略: 1token ≈ 2中文字符)"""
        return len(text) // 2
    
    def _flatten_json(self, data: Any, parent_key: str = "", sep: str = ".") -> List[Dict]:
        """扁平化JSON结构"""
        items = []
        
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(self._flatten_json(v, new_key, sep))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                items.extend(self._flatten_json(item, new_key, sep))
        else:
            items.append({
                "key": parent_key,
                "value": str(data),
                "type": type(data).__name__
            })
        
        return items
    
    def _group_by_semantic(self, flat_items: List[Dict]) -> List[List[Dict]]:
        """按语义分组JSON项"""
        groups = []
        current_group = []
        current_tokens = 0
        
        for item in flat_items:
            # 估算当前项的token数
            item_tokens = self._estimate_tokens(f"{item['key']}: {item['value']}")
            
            # 检查是否需要开始新组
            if (current_tokens + item_tokens > self.max_tokens and 
                current_tokens >= self.min_tokens and current_group):
                
                groups.append(current_group)
                # 保留重叠项
                overlap_items = current_group[-1:] if len(current_group) > 1 else []
                current_group = overlap_items
                current_tokens = sum(self._estimate_tokens(f"{i['key']}: {i['value']}") for i in overlap_items)
            
            current_group.append(item)
            current_tokens += item_tokens
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _format_chunk(self, group: List[Dict]) -> str:
        """格式化JSON块为可读文本"""
        lines = []
        for item in group:
            key = item['key']
            value = item['value']
            item_type = item['type']
            
            # 格式化输出
            if item_type == 'str':
                lines.append(f"{key}: \"{value}\"")
            elif item_type in ['int', 'float']:
                lines.append(f"{key}: {value}")
            elif item_type == 'bool':
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """JSON分块主方法"""
        try:
            # 解析JSON
            json_data = json.loads(text)
        except json.JSONDecodeError as e:
            # 如果不是有效JSON，回退到简单分块
            return self._fallback_chunk(text, doc_id)
        
        print(f"[JsonChunker] 解析JSON成功，开始分块...")
        
        # 扁平化JSON
        flat_items = self._flatten_json(json_data)
        
        if not flat_items:
            return [{"id": f"{doc_id}_0", "doc_id": doc_id, "content": text, "metadata": {"chunk_type": "json_empty"}}]
        
        # 按语义分组
        groups = self._group_by_semantic(flat_items)
        
        # 生成块
        chunks = []
        for i, group in enumerate(groups):
            chunk_content = self._format_chunk(group)
            
            chunks.append({
                "id": f"{doc_id}_json_{i}",
                "doc_id": doc_id,
                "content": chunk_content,
                "metadata": {
                    "chunk_type": "json_semantic",
                    "chunk_index": i,
                    "item_count": len(group),
                    "estimated_tokens": self._estimate_tokens(chunk_content),
                    "keys": [item["key"] for item in group]
                }
            })
        
        print(f"[JsonChunker] JSON分块完成: {len(chunks)} 个块")
        return chunks
    
    def _fallback_chunk(self, text: str, doc_id: str) -> List[Dict]:
        """回退到简单分块"""
        print(f"[JsonChunker] JSON解析失败，使用简单分块...")
        
        chunks = []
        chunk_size = self.max_tokens * 2  # 字符估算
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append({
                "id": f"{doc_id}_fallback_{i}",
                "doc_id": doc_id,
                "content": chunk,
                "metadata": {
                    "chunk_type": "json_fallback",
                    "chunk_index": i // chunk_size
                }
            })
        
        return chunks
