import asyncio

import streamlit as st
from indexing.doc_pipeline import DocumentPipeline
from infra.config import get_config
from retrieve.query_rag import RAGQuery

# Page config
st.set_page_config(page_title="Mine RAG Agent", layout="wide")
st.title("🤖 Mine RAG Agent with DashVector")

# Initialize config
try:
    config = get_config()
except Exception as e:
    st.error(f"❌ Init config failed: {e}")
    st.stop()

# Use config values
collection_name = config.collection_name
dashvector_api_key = config.dashvector_api_key
dashvector_endpoint = config.dashvector_endpoint
top_k = 5


# 分块策略说明
strategy_descriptions = {
    "simple": "简单分块 - 按固定字符数分割",
    "semantic": "语义分块 - 按句子和语义边界分割", 
    "contextual": "上下文分块 - 语义分块 + 上下文增强",
    "intelligent_semantic": "智能语义分块 - AI驱动的语义聚类 + 重叠优化 (推荐)"
}
st.sidebar.caption(f"📋 {strategy_descriptions.get(config.chunking_strategy, '未知策略')}")

# JSON分块说明
st.sidebar.info("🔧 JSON文件自动使用专用分块器，保持数据结构完整性")

# Initialize session state
if 'rag_query' not in st.session_state:
    # Initialize document pipeline and RAG query with config
    st.session_state.doc_pipeline = DocumentPipeline(config)
    st.session_state.rag_query = RAGQuery(config)

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    
    # URL input
    url = st.text_input(
        "Enter URL to load content",
        value="https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
        help="Enter a URL to a PDF or text document"
    )
    
    if st.button("📥 Load from URL"):
        with st.spinner("Loading content..."):
            try:
                # 使用doc_pipeline处理URL内容
                def progress_callback(progress, message):
                    st.info(f"[{progress:.1%}] {message}")
                
                result = asyncio.run(st.session_state.doc_pipeline.process_url_content(
                    url, progress_callback
                ))
                
                if result and result.get("success"):
                    chunks_count = len(result.get("chunks", []))
                    doc_id = result.get("doc_id", "unknown")
                    st.success(f"✅ Content loaded successfully! Processed {chunks_count} chunks")
                    st.info(f"📋 Document ID: `{doc_id}`")
                else:
                    error_msg = result.get("error", "Unknown error") if result else "No result returned"
                    st.error(f"❌ Error loading content: {error_msg}")
                    
            except Exception as e:
                st.error(f"❌ Error loading content: {e}")
    
    # File upload
    st.divider()
    st.subheader("Or upload a file")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md", "json"])
    
    if uploaded_file:
        if st.button("📤 Upload File"):
            with st.spinner("Processing file..."):
                try:
                    # 使用doc_pipeline处理上传文件
                    def progress_callback(progress, message):
                        st.info(f"[{progress:.1%}] {message}")
                    
                    result = asyncio.run(st.session_state.doc_pipeline.process_uploaded_file(
                        uploaded_file, progress_callback
                    ))
                    
                    print(f"[App] Upload result: success={result.get('success')}, chunks={len(result.get('chunks', []))}")
                    
                    if result and result.get("success"):
                        chunks_count = len(result.get("chunks", []))
                        doc_id = result.get("doc_id", "unknown")
                        st.success(f"✅ File uploaded successfully! Processed {chunks_count} chunks")
                        st.info(f"📋 Document ID: `{doc_id}`")
                    else:
                        error_msg = result.get("error", "Unknown error") if result else "No result returned"
                        st.error(f"❌ Error uploading file: {error_msg}")
                        
                except Exception as e:
                    import traceback
                    st.error(f"❌ Error uploading file: {e}")
                    st.error(f"Details: {traceback.format_exc()}")
    
    st.divider()

# Main chat interface
st.subheader("💬 Chat with your documents")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response with RAG query
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                print(f"[Agent] User query: {prompt[:50]}...")
                
                # Use RAG query from query_rag.py
                result = asyncio.run(st.session_state.rag_query.query(prompt, top_k))
                
                print(f"[Agent] RAG query completed")
                st.markdown(result["answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                
            except Exception as e:
                import traceback
                error_msg = f"❌ Error: {e}"
                print(f"[Agent] Traceback: {traceback.format_exc()}")
                st.error(f"Details: {traceback.format_exc()}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
