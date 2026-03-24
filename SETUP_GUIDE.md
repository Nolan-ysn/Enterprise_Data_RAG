# RAG 知识库系统 - 安装与使用指南

## 系统已完成构建 ✅

企业级元数据增强 RAG 知识库系统已成功构建完成！

## 📦 已创建的文件

```
RAG_knowledge/
├── .env                          # 环境变量配置模板
├── .gitignore                    # Git 忽略文件
├── requirements.txt              # Python 依赖列表
├── README.md                     # 完整的项目文档
├── SETUP_GUIDE.md                # 本文件 - 安装指南
├── config.py                     # 配置模块 ✅
├── ingest.py                     # 入库与打标模块 ✅
├── retrieve_qa.py                # 检索与问答模块 ✅
├── main.py                       # CLI 入口 ✅
├── data/
│   ├── raw/.gitkeep             # PDF 文档存放目录
│   └── vectorstore/              # 向量数据库存储目录
└── logs/                         # 日志文件目录
```

## 🔧 环境配置步骤

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 解决 PyTorch 依赖问题（如遇到）

如果遇到 `OSError: [WinError 126]` 错误，这是 PyTorch 在 Windows 上的常见问题。解决方案：

**方案 A：重新安装 PyTorch（推荐）**

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**方案 B：安装 Visual C++ Redistributable**

下载并安装 [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

**方案 C：使用 CPU 版本的 PyTorch**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 配置 API Key

编辑 `.env` 文件，填入您的 API 信息：

```bash
# 硅基流动示例
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_API_KEY=sk-your-actual-api-key-here
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5

# 或使用智谱 GLM
# OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# OPENAI_API_KEY=your-zhipu-api-key
# LLM_MODEL_NAME=glm-4
# EMBEDDING_MODEL_NAME=embedding-v2
```

## 🚀 使用方法

### 1. 查看帮助

```bash
python main.py --help
```

### 2. 入库 PDF 文档

**单个文件入库：**

```bash
python main.py ingest data/raw/report.pdf
```

**批量入库：**

```bash
python main.py batch-ingest data/raw
```

### 3. 提问查询

```bash
python main.py ask "腾讯在人工智能领域有哪些布局？"
```

### 4. 查看系统状态

```bash
python main.py status
```

## 📊 系统特性

### ✅ 已实现的核心功能

1. **元数据增强检索**
   - 自动提取文档中的公司、行业信息
   - 支持基于元数据的精准过滤

2. **结构化输出**
   - 使用 Pydantic V2 + LangChain `with_structured_output`
   - 确保元数据提取的准确性和一致性

3. **意图识别**
   - 智能分析用户问题
   - 自动构建 Metadata 过滤条件

4. **溯源追踪**
   - 所有回答均附带完整的文档溯源信息
   - 显示来源文档、公司、行业等信息

5. **安全可靠**
   - 环境变量管理，绝不硬编码 API Key
   - 完整的异常捕获机制
   - 详细的日志记录

6. **易于扩展**
   - 支持智谱 GLM、硅基流动等多种 OpenAI 兼容 API
   - 模块化设计，易于维护和扩展

## 🏗️ 架构说明

### 核心模块

1. **config.py** - 配置模块
   - 加载环境变量
   - 初始化 LLM 和 Embeddings
   - 配置向量数据库
   - 设置日志系统

2. **ingest.py** - 入库模块
   - PDF 文档加载与切分
   - 使用 LLM 提取元数据（公司、行业）
   - 向量化并存储到 Chroma

3. **retrieve_qa.py** - 检索问答模块
   - 意图识别（提取问题中的公司/行业）
   - Metadata 过滤检索
   - RAG 问答生成

4. **main.py** - CLI 入口
   - 提供命令行接口
   - 支持入库、提问、状态查看等操作

## 📝 代码规范

所有代码严格遵循 `.clinerules` 规范：

- ✅ 禁止使用知识图谱（GraphRAG）
- ✅ 强制使用 Metadata-Augmented RAG 架构
- ✅ 使用 `with_structured_output` + Pydantic V2
- ✅ 所有 API Key 从 `.env` 读取
- ✅ 完整的类型提示和 Docstring
- ✅ 使用 `logging` 模块记录关键节点
- ✅ 所有外部 API 调用包含 `try-except`

## 🐛 常见问题

### Q1: PyTorch 加载失败

**错误：** `OSError: [WinError 126]`

**解决：** 见上文"解决 PyTorch 依赖问题"部分

### Q2: API 调用失败

**错误：** `❌ LLM 初始化失败`

**解决：**
1. 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确
2. 确认 `OPENAI_BASE_URL` 是否可访问
3. 检查网络连接

### Q3: 元数据提取失败

**错误：** `❌ 元数据提取失败`

**解决：**
1. 检查 PDF 文件是否可正常读取
2. 确认 LLM 模型是否支持结构化输出
3. 查看日志文件 `logs/rag_system.log`

### Q4: 检索结果为空

**错误：** `⚠️ 未找到相关文档`

**解决：**
1. 确认已成功入库文档
2. 检查问题中的公司/行业名称是否与元数据匹配
3. 使用 `python main.py status` 查看数据库状态

## 📚 下一步

1. **准备测试数据**
   - 将 PDF 文档放入 `data/raw/` 目录
   - 建议使用包含公司信息的研报或文档

2. **测试入库功能**
   ```bash
   python main.py ingest data/raw/your_document.pdf
   ```

3. **测试问答功能**
   ```bash
   python main.py ask "文档中提到了哪些公司？"
   ```

4. **查看系统状态**
   ```bash
   python main.py status
   ```

## 🎯 系统优势

相比传统 RAG 系统，本系统的优势：

1. **更精准的检索**：通过元数据过滤，减少无关文档的干扰
2. **更好的可解释性**：溯源信息清晰展示答案来源
3. **更高的准确率**：结构化输出确保元数据提取的准确性
4. **更强的扩展性**：模块化设计，易于添加新的元数据字段
5. **更安全的架构**：环境变量管理，完整的错误处理

## 📧 技术支持

如有问题，请：
1. 查看 `logs/rag_system.log` 日志文件
2. 检查 `.env` 配置是否正确
3. 确认所有依赖已正确安装

---

**系统构建完成！祝您使用愉快！** 🎉