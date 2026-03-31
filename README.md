# RAG系统 - DashVector + Ollama

基于DashVector向量数据库和本地Ollama模型的RAG(检索增强生成)系统，支持文档上传、URL内容处理和智能问答。

## 特性

- 🚀 **本地化**: 支持Ollama本地embedding和LLM模型
- 📄 **多格式支持**: PDF、TXT、MD文件上传和URL内容抓取
- 🔍 **智能检索**: 基于DashVector的高效向量搜索
- 💬 **Streamlit界面**: 友好的Web UI交互
- 🔄 **异步处理**: 高性能文档处理和查询
- 📊 **自动维度检测**: 无需手动配置embedding维度

## 快速开始

### 1. 环境配置

创建 `.env` 文件：

```env
# 必需配置
DASHVECTOR_API_KEY=your_dashvector_api_key
DASHVECTOR_ENDPOINT=your_dashvector_endpoint
COLLECTION_NAME=rag_experiment

# Ollama配置 (可选，有默认值)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# OpenAI配置 (可选，如果使用OpenAI)
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-3.5-turbo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动应用

```bash
streamlit run app.py
```

### 4. 使用Ollama (推荐)

确保本地运行Ollama：
```bash
# 安装embedding模型
ollama pull mxbai-embed-large

# 安装LLM模型 (可选)
ollama pull llama2
```

## 项目架构

```
rag-experiment/
├── app.py                    # Streamlit主应用
├── requirements.txt           # 项目依赖
├── .env                     # 环境配置
├── infra/                   # 基础设施层
│   ├── config.py           # 配置管理
│   ├── llm.py              # LLM封装
│   └── dashvector_db.py    # DashVector集成
├── indexing/               # 文档处理层
│   ├── embedding.py        # Embedding模型 (LangChain + Ollama)
│   ├── doc_chunk.py        # 文档分块
│   ├── doc_uploader.py     # 文档上传
│   ├── doc_pipeline.py     # 文档处理流水线
│   └── store_manager.py    # 向量存储管理
├── retrieve/               # 检索层
│   └── query_rag.py        # RAG查询逻辑
└── deepdoc/               # 文档解析 (可选)
    └── parser/            # 各种格式解析器
```

## 核心组件

### EmbeddingModel
- 使用LangChain + Ollama本地embedding
- 支持多种模型: `mxbai-embed-large`, `all-minilm`, `nomic-embed-text`
- 自动维度检测

### RAGQuery
- 完整的RAG查询流程
- 异步文档检索
- 智能上下文构建

### DocumentPipeline
- 文档上传和URL处理
- 自动分块和embedding
- 进度回调支持

## 使用示例

### 编程接口

```python
from retrieve.query_rag import RAGQuery
from infra.config import Config

# 初始化
config = Config().validate()
rag = RAGQuery(config)

# 查询
result = await rag.query("什么是机器学习？", top_k=3)
print(result["answer"])
print(f"引用文档数: {len(result['sources'])}")
```

### Streamlit界面

1. **文档上传**: 支持PDF、TXT、MD文件
2. **URL处理**: 自动抓取网页内容
3. **智能问答**: 基于检索内容的回答
4. **实时反馈**: 处理进度和结果展示

### 智能语义分块
1. 四种分块策略
Simple: 基础固定字符分块
Semantic: 句子边界语义分块
Contextual: 上下文增强分块
Intelligent Semantic: AI驱动的语义聚类分块 ⭐
2. 智能语义分块特性
🔍 句子分割 → 🧠 向量生成 → 🎯 语义聚类 → 📦 块合并 → 🔗 重叠优化
3. 核心算法
句子分割: 中英文混合模式
向量生成: 批量处理，API限制保护
语义聚类: 基于余弦相似度 (阈值0.7)
智能合并: 50-300字符块大小控制
重叠优化: 75 tokens重叠避免上下文断裂
🚀 技术优势
相比传统分块:
✅ 语义完整性: 相似句子聚在一起
✅ 上下文连贯: 重叠避免语义断裂
✅ 自适应大小: 根据内容动态调整
✅ 高质量检索: 更精准的语义匹配
处理流程:
文档 (6048字符)
    ↓
句子分割 (约120个句子)
    ↓
向量生成 (120个1024维向量)
    ↓
语义聚类 (15个语义簇)
    ↓
智能合并 (12-18个块)
    ↓
重叠优化 (最终输出)

## 配置说明

### Ollama模型推荐

| 模型 | 维度 | 大小 | 特点 |
|------|------|------|------|
| `mxbai-embed-large` | 1024 | 670MB | 平衡性能 |
| `all-minilm` | 384 | 90MB | 轻量快速 |
| `nomic-embed-text` | 768 | 270MB | 高质量 |

### DashVector配置

- **API Key**: 从阿里云DashVector控制台获取
- **Endpoint**: 集群访问地址
- **Collection**: 向量集合名称

## 故障排除

### 常见问题

1. **Ollama连接失败**
   ```bash
   # 检查Ollama服务
   ollama list
   curl http://localhost:11434/api/tags
   ```

2. **DashVector连接错误**
   - 检查API Key和Endpoint
   - 确认网络连接

3. **Embedding维度不匹配**
   - 系统会自动检测维度
   - 如需手动设置，删除现有collection重新创建

## 许可证

MIT License
