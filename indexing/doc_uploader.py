import asyncio
import tempfile
import os
import uuid
from typing import List, Dict, Optional, Callable
from indexing.doc_chunk import DocumentProcessor
from infra.config import get_config

def get_uuid():
    """生成UUID"""
    return str(uuid.uuid4())

class DocumentUploader:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.processor = DocumentProcessor(self.config)
    
    async def process_uploaded_file(self, uploaded_file, 
                                   progress_callback: Optional[Callable] = None) -> Dict:
        """处理上传的文件"""
        try:
            print(f"[DocumentUploader] Processing file: {uploaded_file.name}")
            
            if progress_callback:
                progress_callback(0.1, "保存文件...")
            
            # 保存上传的文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_suffix(uploaded_file.name)) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            print(f"[DocumentUploader] File saved to: {file_path}")
            
            if progress_callback:
                progress_callback(0.3, "提取文本...")
            
            # 提取文本内容
            text_content = self._extract_text(file_path, uploaded_file.name)
            print(f"[DocumentUploader] Extracted text length: {len(text_content)}")
            
            if not text_content.strip():
                raise ValueError("提取的文本内容为空")
            
            if progress_callback:
                progress_callback(0.5, "处理文档...")
            
            # 处理文档（分块和embedding）
            doc_id = uploaded_file.id if hasattr(uploaded_file, "id") else get_uuid()
            chunks = await self.processor.process_document(
                text_content, 
                doc_id=doc_id,
                file_name=uploaded_file.name,
                progress_callback=progress_callback
            )
            
            print(f"[DocumentUploader] Processed {len(chunks)} chunks")
            
            # 清理临时文件
            os.unlink(file_path)
            
            if progress_callback:
                progress_callback(1.0, f"处理完成，共{len(chunks)}个文档块")
            
            return {
                "success": True,
                "chunks": chunks,
                "file_name": uploaded_file.name,
                "doc_id": doc_id,
                "content_length": len(text_content)
            }
            
        except Exception as e:
            print(f"[DocumentUploader] Error processing file: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "file_name": uploaded_file.name if uploaded_file else "unknown",
                "doc_id": None
            }
    
    async def process_url_content(self, url: str, 
                                progress_callback: Optional[Callable] = None) -> Dict:
        """处理URL内容"""
        try:
            if progress_callback:
                progress_callback(0.1, f"获取URL内容: {url}")
            
            # 获取URL内容
            text_content = await self._fetch_url_content(url)
            
            if progress_callback:
                progress_callback(0.5, "处理文档...")
            
            # 处理文档（分块和embedding）
            # 为URL生成基于内容的doc_id
            url_base = url.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '_')
            doc_id = f"url_{url_base}_{get_uuid()[:8]}"
            chunks = await self.processor.process_document(
                text_content, 
                doc_id=doc_id,
                file_name=uploaded_file.name,
                progress_callback=progress_callback
            )
            
            if progress_callback:
                progress_callback(1.0, f"处理完成，共{len(chunks)}个文档块")
            
            return {
                "success": True,
                "chunks": chunks,
                "url": url,
                "doc_id": doc_id,
                "content_length": len(text_content)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "doc_id": None
            }
    
    def _get_file_suffix(self, filename: str) -> str:
        """获取文件后缀"""
        return os.path.splitext(filename)[1].lower() if filename else '.txt'
    
    def _extract_text(self, file_path: str, filename: str) -> str:
        """提取文件文本内容"""
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == '.json':
            # 处理JSON文件
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 将JSON转换为可读文本
                    return self._json_to_text(data)
            except Exception as e:
                raise ImportError(f"JSON parse error: {e}")
        elif file_extension == '.pdf':
            # 需要安装PyPDF2或pdfplumber
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise ImportError("需要安装PyPDF2: pip install PyPDF2")
        else:
            # 尝试作为文本文件读取
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _json_to_text(self, data) -> str:
        """将JSON数据转换为可读文本"""
        import json
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text_parts.append(f"{key}:\n{self._json_to_text(value)}")
                else:
                    text_parts.append(f"{key}: {value}")
            return "\n".join(text_parts)
        elif isinstance(data, list):
            text_parts = []
            for i, item in enumerate(data):
                text_parts.append(f"Item {i+1}:\n{self._json_to_text(item)}")
            return "\n".join(text_parts)
        else:
            return str(data)
    
    async def _fetch_url_content(self, url: str) -> str:
        """获取URL内容"""
        try:
            import aiohttp
            import aiofiles
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        raise Exception(f"HTTP {response.status}: Can not fetch URL content")
                        
        except ImportError:
            # 如果没有aiohttp，使用requests的同步版本
            import requests
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
            else:
                raise Exception(f"HTTP {response.status_code}: Can not fetch URL content")
