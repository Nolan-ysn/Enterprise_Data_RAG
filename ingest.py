"""
文档入库与元数据提取模块
负责读取 PDF 文件、切分文本、提取元数据并存入向量数据库
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from config import Config, get_llm, get_vectorstore, logger


class DocMetadata(BaseModel):
    """
    文档元数据提取模型
    使用 Pydantic V2 定义结构化输出格式
    """
    companies: List[str] = Field(
        description="文档中提到的公司名称列表，提取所有相关公司"
    )
    industry: str = Field(
        description="文档所属的行业领域，如：互联网、金融、医疗、制造业等"
    )


def extract_metadata_from_chunks(chunks: List[str]) -> DocMetadata:
    """
    从文本块中提取元数据
    
    Args:
        chunks: 文本块列表
        
    Returns:
        DocMetadata: 提取的元数据对象
    """
    try:
        llm = get_llm(temperature=0.0)
        
        # 截取前 5 个文本块用于元数据提取
        sample_text = "\n\n".join(chunks[:5])
        
        # 使用 with_structured_output 进行结构化输出
        structured_llm = llm.with_structured_output(DocMetadata)
        
        logger.info("🔍 开始提取文档元数据...")
        metadata = structured_llm.invoke(
            f"""请分析以下文档内容，提取其中的公司名称和所属行业。

重要注意事项：
1. 请忽略文档开头的水印信息（如"华安证券股份有限公司"、"中信证券股份有限公司"等券商名称）
2. 只提取文档中实际分析的公司，不要提取发布研报的券商
【实体标准化】：提取公司名称时，必须提取核心品牌名！强制去除所有股票代码后缀(如“-SW”、“-W”、“-U”)以及企业性质后缀(如“股份有限公司”、“集团”)。
3. 重点关注文档正文内容，而非页眉页脚的水印
4. 如果文档分析的是某家上市公司，请提取该上市公司的名称
5.【关键】请严格按照以下 JSON 格式输出，不要添加任何其他文字或解释：
   {{"companies": ["公司1", "公司2"], "industry": "行业名称"}}

文档内容：
{sample_text}"""
        )
        
        logger.info(f"✅ 元数据提取成功 - 公司: {metadata.companies}, 行业: {metadata.industry}")
        return metadata
        
    except Exception as e:
        logger.error(f"❌ 元数据提取失败: {e}")
        # 返回默认元数据
        return DocMetadata(companies=["未知"], industry="未知")


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """
    加载 PDF 文件并进行文本切分
    
    Args:
        pdf_path: PDF 文件路径
        
    Returns:
        List[Document]: 切分后的文档块列表
    """
    try:
        logger.info(f"📄 开始加载 PDF 文件: {pdf_path}")
        
        # 加载 PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"✅ PDF 加载成功，共 {len(documents)} 页")
        
        # 文本切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"✅ 文本切分完成，共 {len(chunks)} 个文本块")
        
        return chunks
        
    except Exception as e:
        logger.error(f"❌ PDF 加载或切分失败: {e}")
        raise


def ingest_document(pdf_path: str) -> Dict[str, Any]:
    """
    将 PDF 文档入库到向量数据库
    
    Args:
        pdf_path: PDF 文件路径
        
    Returns:
        Dict[str, Any]: 入库结果统计信息
    """
    try:
        # 验证文件存在
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        
        # 加载并切分 PDF
        chunks = load_and_split_pdf(pdf_path)
        
        # 提取元数据
        chunk_texts = [chunk.page_content for chunk in chunks]
        metadata = extract_metadata_from_chunks(chunk_texts)
        if not metadata.companies:
            metadata.companies = ["未知"]
            
        companies_str = ",".join(metadata.companies) 
        # 为每个文档块添加元数据
        for chunk in chunks:
            chunk.metadata.update({
                "companies": metadata.companies,
                "industry": metadata.industry,
                "source": pdf_file.name
            })
        
        # 存入向量数据库
        logger.info("💾 开始将文档存入向量数据库...")
        vectorstore = get_vectorstore()
        
        # 添加文档到向量数据库
        vectorstore.add_documents(chunks)
        
        # 获取当前向量数据库中的文档总数
        collection = vectorstore._collection
        total_docs = collection.count()
        
        result = {
            "success": True,
            "pdf_path": pdf_path,
            "chunks_count": len(chunks),
            "companies": metadata.companies,
            "industry": metadata.industry,
            "total_docs_in_db": total_docs
        }
        
        logger.info(f"✅ 文档入库成功！")
        logger.info(f"   - 文档块数量: {len(chunks)}")
        logger.info(f"   - 提取公司: {metadata.companies}")
        logger.info(f"   - 所属行业: {metadata.industry}")
        logger.info(f"   - 数据库总文档数: {total_docs}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 文档入库失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "pdf_path": pdf_path
        }


def batch_ingest(directory: str) -> List[Dict[str, Any]]:
    """
    批量入库目录下的所有 PDF 文件
    
    Args:
        directory: PDF 文件所在目录
        
    Returns:
        List[Dict[str, Any]]: 每个文件的入库结果列表
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        # 查找所有 PDF 文件
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️ 目录中没有找到 PDF 文件: {directory}")
            return []
        
        logger.info(f"📁 找到 {len(pdf_files)} 个 PDF 文件，开始批量入库...")
        
        results = []
        for pdf_file in pdf_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"处理文件: {pdf_file.name}")
            logger.info(f"{'='*60}")
            result = ingest_document(str(pdf_file))
            results.append(result)
        
        # 统计结果
        success_count = sum(1 for r in results if r.get("success", False))
        logger.info(f"\n{'='*60}")
        logger.info(f"批量入库完成！成功: {success_count}/{len(results)}")
        logger.info(f"{'='*60}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 批量入库失败: {e}")
        return []


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python ingest.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    result = ingest_document(pdf_path)
    
    if result["success"]:
        print("\n✅ 入库成功！")
        print(f"文档块数量: {result['chunks_count']}")
        print(f"提取公司: {result['companies']}")
        print(f"所属行业: {result['industry']}")
    else:
        print(f"\n❌ 入库失败: {result.get('error', '未知错误')}")