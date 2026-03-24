"""
检索与问答模块
负责意图识别、Metadata 过滤检索和 RAG 问答
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import Config, get_llm, get_vectorstore, logger
from pydantic import BaseModel, Field
import json


class IntentSchema(BaseModel):
    """用户意图的结构化输出定义"""
    companies: List[str] = Field(default_factory=list, description="问题中提到的公司名称列表（如无则为空列表）")
    industry: str = Field(default="", description="问题中提到的行业关键词（如无则为空字符串）")


class IntentAnalyzer:
    """意图识别器"""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个精准的金融意图识别系统。你的任务是从用户的提问中，提取出用于数据库检索的实体标签。
            
【提取规则】：
1. companies (公司列表)：提取问题中所有明确提到的具体企业、品牌或机构名称。即使只提到简称，也必须提取。
             【极其重要】：必须进行实体标准化，强制去掉所有的股票后缀(如“-SW”)和公司后缀。例如用户问“蔚来-SW的盈利”,你必须提取为“蔚来”。
             如果没有提到具体公司，必须返回空列表 []。
2. industry (行业名称)：提取问题中提到的宏观行业或赛道领域（例如："新能源汽车"、"半导体"）。如果没有提到，必须返回空字符串 ""。

请仔细分析用户的提问，绝不能遗漏问题中的实体名词！"""),
            ("human", "用户问题：{question}")
        ])
        
        # 绑定结构化输出
        self.structured_llm = self.llm.with_structured_output(IntentSchema)
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """
        分析用户问题的意图
        
        Args:
            question: 用户问题
            
        Returns:
            Dict[str, Any]: 包含 companies 和 industry 的字典
        """
        try:
            logger.info(f"🔍 分析用户问题意图: {question}")
            
            # 使用 LLM 提取意图
            chain = self.prompt | self.structured_llm
            result = chain.invoke({"question": question})
            
            # 解析结果（简单处理，实际可以使用更复杂的 JSON 解析）
            import json
            intent_dict = result.model_dump()
            logger.info(f"✅ 意图识别成功 - 公司: {intent_dict.get('companies',[])}, 行业: {intent_dict.get('industry', '')}")
            return intent_dict
                
        except Exception as e:
            logger.error(f"❌ 意图识别失败: {e}")
            return {"companies": [], "industry": ""}


class MetadataFilterRetriever:
    """基于 Metadata 过滤的检索器"""
    
    def __init__(self):
        self.vectorstore = get_vectorstore()
        self.intent_analyzer = IntentAnalyzer()
    
    def build_filter(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据意图构建 Metadata 过滤条件
        
        Args:
            intent: 意图分析结果
            
        Returns:
            Dict[str, Any]: Chroma 过滤条件
        """
        filters = {}
        
        companies = intent.get("companies",[])
        if companies and len(companies) > 0:
            # 🚨 致命关键点：取出列表里的第一个元素 '蔚来'，强转为字符串！
            company_name = str(companies[0]) 
            # 传给 Chroma 的必须是纯字符串 company_name，绝对不能是 companies 列表！
            filters["companies"] = {"$contains": company_name}
        
        # 🌟 行业过滤同样做绝对安全处理
        industry = intent.get("industry", "")
        if industry:
            filters["industry"] = {"$contains": str(industry)}
        
        return filters
    def retrieve(self, question: str, k: Optional[int] = None) -> List[Any]:
        """
        检索相关文档
        
        Args:
            question: 用户问题
            k: 返回的文档数量，默认使用配置值
            
        Returns:
            List[Any]: 检索到的文档列表
        """
        try:
            # 分析意图
            intent = self.intent_analyzer.analyze(question)
            
            # 构建过滤条件
            filters = self.build_filter(intent)
            
            # 设置检索数量
            k = k or Config.RETRIEVER_K
            
            logger.info(f"🔎 开始检索相关文档...")
            logger.info(f"   - 过滤条件: {filters}")
            logger.info(f"   - 返回数量: {k}")
            
            # 执行检索
            if filters:
                # 带过滤条件的检索
                docs = self.vectorstore.similarity_search(
                    question,
                    k=k,
                    filter=filters
                )
            else:
                # 无过滤条件的检索
                docs = self.vectorstore.similarity_search(question, k=k)
            
            logger.info(f"✅ 检索完成，找到 {len(docs)} 个相关chunk")
            
            return docs
            
        except Exception as e:
            logger.error(f"❌ 检索失败: {e}")
            return []


class RAGChain:
    """RAG 问答链"""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.3)
        self.retriever = MetadataFilterRetriever()
        
        # 构建提示词模板
        self.prompt = ChatPromptTemplate.from_template(
            """你是一个专业的知识库助手。请根据以下参考文档回答用户的问题。

参考文档：
{context}

用户问题：{question}

回答要求：
1. 基于参考文档内容回答，不要编造信息
2. 如果参考文档中没有相关信息，请明确说明
3. 回答要清晰、准确、有条理

现在请回答用户的问题："""
        )
    
    def format_docs(self, docs: List[Any]) -> str:
        """格式化检索到的文档（合并相同来源）"""
        # 按来源分组
        source_groups = {}
        for doc in docs:
            source = doc.metadata.get("source", "未知")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        formatted_docs = []
        for i, (source, group_docs) in enumerate(source_groups.items(), 1):
            companies = group_docs[0].metadata.get("companies", [])
            industry = group_docs[0].metadata.get("industry", "未知")
            
            # 合并相同来源的内容
            combined_content = "\n\n".join([doc.page_content for doc in group_docs])
            
            formatted = f"""
    【文档 {i}】
    来源：{source}
    公司：{', '.join(companies)}
    行业：{industry}
    内容：{combined_content}
    """
            formatted_docs.append(formatted)
        
        return "\n".join(formatted_docs)

    
    def get_source_info(self, docs: List[Any]) -> str:
        """
        获取溯源信息
        
        Args:
            docs: 文档列表
            
        Returns:
            str: 溯源信息文本
        """
        unique_sources = {}
        for doc in docs:
            source = doc.metadata.get("source", "未知")
            companies = doc.metadata.get("companies", [])
            industry = doc.metadata.get("industry", "未知")
            
            # 如果这个文档还没有出现过，添加到字典
            if source not in unique_sources:
                unique_sources[source] = {
                    "companies": companies,
                    "industry": industry
                }
        
            # 生成去重后的溯源信息
        sources = []
        for source, info in unique_sources.items():
            sources.append(f"- 来源：{source}（公司：{', '.join(info['companies'])}，行业：{info['industry']}）")
            
        return "\n".join(sources)
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        回答用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            Dict[str, Any]: 包含答案和溯源信息的字典
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"用户问题: {question}")
            logger.info(f"{'='*60}")
            
            # 检索相关文档
            docs = self.retriever.retrieve(question)
            
            if not docs:
                logger.warning("⚠️ 未找到相关chunk")
                return {
                    "success": True,
                    "answer": "抱歉，知识库中没有找到与您问题相关的信息。",
                    "sources": [],
                    "docs_count": 0
                }
            
            # 格式化文档
            context = self.format_docs(docs)
            
            # 构建问答链
            chain = (
                {
                    "context": lambda x: self.format_docs(x["docs"]),
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            # 执行问答
            logger.info("🤖 生成回答...")
            answer = chain.invoke({"docs": docs, "question": question})
            
            # 获取溯源信息
            sources = self.get_source_info(docs)
            
            # 组装最终答案（包含溯源信息）
            final_answer = f"{answer}\n\n【溯源信息】\n{sources}"
            
            logger.info(f"✅ 回答生成成功")
            logger.info(f"   - 引用chunk数: {len(docs)}")
            
            return {
                "success": True,
                "answer": final_answer,
                "sources": sources,
                "docs_count": len(docs),
                "docs": docs
            }
            
        except Exception as e:
            logger.error(f"❌ 问答失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "抱歉，回答问题时出现错误，请稍后重试。"
            }


def ask_question(question: str) -> Dict[str, Any]:
    """
    便捷函数：回答用户问题
    
    Args:
        question: 用户问题
        
    Returns:
        Dict[str, Any]: 问答结果
    """
    rag_chain = RAGChain()
    return rag_chain.ask(question)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python retrieve_qa.py \"您的问题\"")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    result = ask_question(question)
    
    if result["success"]:
        print("\n" + "="*60)
        print("回答：")
        print("="*60)
        print(result["answer"])
        print("\n" + "="*60)
        print(f"引用chunk数: {result['docs_count']}")
        print("="*60)
    else:
        print(f"\n❌ 问答失败: {result.get('error', '未知错误')}")