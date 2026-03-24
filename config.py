"""
统一配置模块
负责加载环境变量并初始化 LangChain 核心组件
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# 加载环境变量
load_dotenv()


class Config:
    """系统配置类"""
    
    # API 配置
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # 模型配置
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
    
    # 向量数据库配置
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/vectorstore")
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/rag_system.log")
    
    # 文本分割配置
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # 检索配置
    RETRIEVER_K: int = 4  # 检索返回的文档数量
    
    @classmethod
    def validate(cls) -> bool:
        """
        验证配置是否完整
        
        Returns:
            bool: 配置是否有效
        """
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your_api_key_here":
            logging.error("❌ 请在 .env 文件中设置有效的 OPENAI_API_KEY")
            return False
        return True


def setup_logging() -> logging.Logger:
    """
    配置日志系统
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录
    log_dir = Path(Config.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    获取大语言模型实例
    
    Args:
        temperature: 温度参数，控制输出的随机性
        
    Returns:
        ChatOpenAI: 配置好的大语言模型实例
    """
    try:
        llm = ChatOpenAI(
            model=Config.LLM_MODEL_NAME,
            base_url=Config.OPENAI_BASE_URL,
            api_key=Config.OPENAI_API_KEY,
            temperature=temperature,
            timeout=60
        )
        logging.info(f"✅ LLM 初始化成功: {Config.LLM_MODEL_NAME}")
        return llm
    except Exception as e:
        logging.error(f"❌ LLM 初始化失败: {e}")
        raise


def get_embeddings() -> OpenAIEmbeddings:
    """
    获取嵌入模型实例
    
    Returns:
        OpenAIEmbeddings: 配置好的嵌入模型实例
    """
    try:
        embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL_NAME,
            base_url=Config.OPENAI_BASE_URL,
            api_key=Config.OPENAI_API_KEY,
            timeout=60
        )
        logging.info(f"✅ Embeddings 初始化成功: {Config.EMBEDDING_MODEL_NAME}")
        return embeddings
    except Exception as e:
        logging.error(f"❌ Embeddings 初始化失败: {e}")
        raise


def get_vectorstore() -> Chroma:
    """
    获取或创建向量数据库实例
    
    Returns:
        Chroma: 配置好的向量数据库实例
    """
    try:
        # 确保持久化目录存在
        persist_dir = Path(Config.CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings = get_embeddings()
        
        # 尝试加载已存在的向量数据库
        if (persist_dir / "chroma.sqlite3").exists():
            vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
                collection_name="rag_documents"
            )
            logging.info(f"✅ 向量数据库加载成功: {Config.CHROMA_PERSIST_DIR}")
        else:
            # 创建新的向量数据库
            vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
                collection_name="rag_documents"
            )
            logging.info(f"✅ 向量数据库创建成功: {Config.CHROMA_PERSIST_DIR}")
        
        return vectorstore
    except Exception as e:
        logging.error(f"❌ 向量数据库初始化失败: {e}")
        raise


def clear_vectorstore() -> bool:
    """
    清空向量数据库中的所有文档
    
    Returns:
        bool: 是否清空成功
    """
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        
        # 获取当前文档数量
        count_before = collection.count()
        
        if count_before == 0:
            logger.info("⚠️ 向量数据库已经是空的")
            return True
        
        # 删除所有文档
        # 获取所有文档的 ID
        ids = collection.get()["ids"]
        # 删除这些文档
        collection.delete(ids=ids)

        
        logger.info(f"✅ 向量数据库已清空，删除了 {count_before} 个文档")
        return True
        
    except Exception as e:
        logger.error(f"❌ 清空向量数据库失败: {e}")
        return False


# 初始化日志
logger = setup_logging()
