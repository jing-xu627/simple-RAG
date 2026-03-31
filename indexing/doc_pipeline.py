import asyncio
import time
import tempfile
import os
from typing import List, Dict, Optional, Callable
from infra.config import get_config
from indexing.doc_uploader import DocumentUploader
from indexing.store_manager import VectorStoreManager


class DocumentPipeline:
    """完整的文档处理流水线"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.uploader = DocumentUploader(self.config)
        self.store_manager = VectorStoreManager(self.config)
    
    async def process_uploaded_file(self, uploaded_file, 
                                   progress_callback: Optional[Callable] = None) -> Dict:
        """处理上传文件的完整流程"""
        try:
            # 1. 处理文件（分块+embedding）
            result = await self.uploader.process_uploaded_file(uploaded_file, progress_callback)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "chunks_count": 0,
                    "doc_id": None,
                    "chunks": []
                }
            
            chunks = result["chunks"]
            
            # 2. 存储到向量数据库
            success = await self.store_manager.store_chunks(chunks, progress_callback)
            
            return {
                "success": success,
                "chunks_count": len(chunks),
                "doc_id": chunks[0]["doc_id"] if chunks else None,
                "chunks": chunks
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "chunks_count": 0,
                "doc_id": None,
                "chunks": []
            }
    
    async def process_url_content(self, url: str, 
                                 progress_callback: Optional[Callable] = None) -> Dict:
        """处理URL内容的完整流程"""
        try:
            # 1. 处理URL内容（分块+embedding）
            result = await self.uploader.process_url_content(url, progress_callback)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "chunks_count": 0,
                    "doc_id": None,
                    "chunks": []
                }
            
            chunks = result["chunks"]
            
            # 2. 存储到向量数据库
            success = await self.store_manager.store_chunks(chunks, progress_callback)
            
            return {
                "success": success,
                "chunks_count": len(chunks),
                "doc_id": chunks[0]["doc_id"] if chunks else None,
                "chunks": chunks
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "chunks_count": 0,
                "doc_id": None,
                "chunks": []
            }

# 便捷函数
async def process_document_upload(uploaded_file, config: Optional[Config] = None,
                                progress_callback: Optional[Callable] = None) -> Dict:
    """便捷的文档上传处理函数"""
    pipeline = DocumentPipeline(config)
    return await pipeline.process_uploaded_file(uploaded_file, progress_callback)

async def process_document_url(url: str, config: Optional[Config] = None,
                              progress_callback: Optional[Callable] = None) -> Dict:
    """便捷的URL文档处理函数"""
    pipeline = DocumentPipeline(config)
    return await pipeline.process_url_content(url, progress_callback)
