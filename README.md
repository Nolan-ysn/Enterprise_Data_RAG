# 企业级元数据增强 RAG 知识库系统

基于 LangChain 构建的轻量级、高准确率 RAG 系统，采用【大模型预提取 Metadata + 标量过滤与向量混合检索】架构。

## ✨ 核心特性

- 🎯 **元数据增强检索**：自动提取文档中的公司、行业等元数据，支持精准过滤
- 🤖 **结构化输出**：使用 Pydantic V2 + LangChain `with_structured_output` 确保元数据提取准确
- 🔍 **意图识别**：智能分析用户问题，自动构建 Metadata 过滤条件
- 📚 **溯源追踪**：所有回答均附带完整的文档溯源信息
- 🔒 **安全可靠**：环境变量管理、完整异常捕获、详细日志记录
- 🚀 **易于扩展**：支持智谱 GLM、硅基流动等多种 OpenAI 兼容 API

## 🏗️ 系统架构

```
┌─────────────┐
│   PDF 文档   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  文档切分 (Chunking)   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  元数据提取 (LLM)       │
│  - companies: List[str] │
│  - industry: str        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  向量化 + 存储到 Chroma  │
└─────────────────────────┘

┌─────────────┐
│  用户问题    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  意图识别 (LLM)          │
│  - 提取公司/行业关键词   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Metadata 过滤检索       │
│  - 向量相似度匹配        │
│  - 标量条件过滤          │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  RAG 问答 (LLM)          │
│  - 生成回答              │
│  - 附带溯源信息          │
└─────────────────────────┘
```

## 📦 安装部署

### 1. 环境要求

- Python 3.9+
- pip

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env` 文件并配置您的 API 信息：

```bash
# 编辑 .env 文件
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_API_KEY=your_api_key_here
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5
```

**支持的 API 服务商：**

- 硅基流动：`https://api.siliconflow.cn/v1`
- 智谱 GLM：`https://open.bigmodel.cn/api/paas/v4`
- 其他 OpenAI 兼容接口

## 🚀 快速开始

### 1. 文档入库

**入库单个 PDF 文档：**

```bash
python main.py ingest data/raw/report.pdf
```

**批量入库目录下的所有 PDF：**

```bash
python main.py batch-ingest data/raw
```

### 2. 提问查询

```bash
python main.py ask "腾讯在人工智能领域有哪些布局？"
```

### 3. 查看系统状态

```bash
python main.py status
```

## 📁 项目结构

```
RAG_knowledge/
├── .env                          # 环境变量配置
├── .gitignore                    # Git 忽略文件
├── requirements.txt              # Python 依赖
├── README.md                     # 项目说明
├── config.py                     # 配置模块
├── ingest.py                     # 入库与打标模块
├── retrieve_qa.py                # 检索与问答模块
├── main.py                       # CLI 入口
├── data/
│   ├── raw/                      # 原始 PDF 文档
│   └── vectorstore/              # Chroma 向量数据库
└── logs/                         # 日志文件
```

## 🔧 核心模块说明

### config.py - 配置模块

- 加载环境变量
- 初始化 LLM 和 Embeddings 模型
- 配置向量数据库连接
- 设置日志系统

### ingest.py - 入库模块

**核心功能：**

1. **文档加载与切分**：使用 `PyPDFLoader` 加载 PDF，`RecursiveCharacterTextSplitter` 切分文本
2. **元数据提取**：使用 `with_structured_output` + Pydantic V2 提取公司、行业信息
3. **向量存储**：将文档块和元数据存入 Chroma 向量数据库

**关键代码：**

```python
class DocMetadata(BaseModel):
    companies: List[str] = Field(description="文档中提到的公司名称列表")
    industry: str = Field(description="文档所属的行业领域")

# 使用结构化输出提取元数据
structured_llm = llm.with_structured_output(DocMetadata)
metadata = structured_llm.invoke(f"分析文档：{sample_text}")
```

### retrieve_qa.py - 检索问答模块

**核心功能：**

1. **意图识别**：分析用户问题，提取公司、行业关键词
2. **Metadata 过滤检索**：根据意图构建过滤条件，执行混合检索
3. **RAG 问答**：基于检索结果生成回答，附带溯源信息

**关键代码：**

```python
# 构建 Metadata 过滤条件
filters = {"companies": {"$in": ["腾讯"]}}

# 执行过滤检索
docs = vectorstore.similarity_search(
    question,
    k=4,
    filter=filters
)
```

### main.py - CLI 入口

提供四个命令：

- `ingest <pdf_path>` - 入库单个 PDF
- `batch-ingest <directory>` - 批量入库
- `ask <question>` - 提问
- `status` - 查看系统状态

## 📊 使用示例

### 示例 1：入库文档

```bash
$ python main.py ingest data/raw/tencent_report.pdf

============================================================
✅ 文档入库成功！
============================================================
📄 文档路径: data/raw/tencent_report.pdf
📊 文档块数量: 45
🏢 提取公司: 腾讯, 微信, 腾讯云
🏭 所属行业: 互联网
💾 数据库总文档数: 45
============================================================
```

### 示例 2：提问查询

```bash
$ python main.py ask "腾讯在人工智能领域有哪些布局？"

============================================================
🤖 回答
============================================================
根据文档内容，腾讯在人工智能领域的主要布局包括：

1. **基础研究**：建立腾讯 AI Lab，专注于机器学习、计算机视觉等前沿技术研究
2. **产品应用**：将 AI 技术应用于微信、QQ等核心产品
3. **云服务**：通过腾讯云提供 AI 能力输出
4. **投资布局**：投资多家 AI 初创公司

【溯源信息】
- 来源：tencent_report.pdf（公司：腾讯, 微信, 腾讯云，行业：互联网）
- 来源：tencent_report.pdf（公司：腾讯, 微信, 腾讯云，行业：互联网）

============================================================
📚 引用文档数: 2
============================================================
```

### 示例 3：查看系统状态

```bash
$ python main.py status

============================================================
📊 RAG 知识库系统状态
============================================================

🔧 配置信息:
  - LLM 模型: Qwen/Qwen2.5-7B-Instruct
  - Embedding 模型: BAAI/bge-large-zh-v1.5
  - API Base URL: https://api.siliconflow.cn/v1
  - 向量数据库路径: ./data/vectorstore

💾 向量数据库:
  - 文档总数: 45
  - 涉及公司数: 3
  - 涉及行业数: 1
  - 文档来源数: 1

🏢 公司列表:
  - 腾讯
  - 微信
  - 腾讯云

🏭 行业列表:
  - 互联网

📄 文档列表:
  - tencent_report.pdf

============================================================
```

## 🔒 安全规范

- ✅ 所有 API Key 从 `.env` 文件读取，绝不硬编码
- ✅ 使用 `with_structured_output` + Pydantic V2 进行结构化输出
- ✅ 所有函数包含完整的类型提示和 Docstring
- ✅ 使用 `logging` 模块记录关键业务节点
- ✅ 所有外部 API 调用包含 `try-except` 容错机制

## 📝 开发规范

遵循 `.clinerules` 目录中的规范：

- **架构底线**：禁止使用知识图谱（GraphRAG），强制使用 Metadata-Augmented RAG
- **编码规范**：结构化输出、安全第一、工程素养
- **工作流**：Plan Before Code、模块化拆分

## 🐛 故障排查

### 问题 1：API 调用失败

**错误信息：** `❌ LLM 初始化失败`

**解决方案：**
1. 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确
2. 确认 `OPENAI_BASE_URL` 是否可访问
3. 检查网络连接

### 问题 2：元数据提取失败

**错误信息：** `❌ 元数据提取失败`

**解决方案：**
1. 检查 PDF 文件是否可正常读取
2. 确认 LLM 模型是否支持结构化输出
3. 查看日志文件 `logs/rag_system.log` 获取详细错误信息

### 问题 3：检索结果为空

**错误信息：** `⚠️ 未找到相关文档`

**解决方案：**
1. 确认已成功入库文档
2. 检查问题中的公司/行业名称是否与元数据匹配
3. 使用 `python main.py status` 查看数据库状态

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。