"""
RAG 知识库系统 - CLI 入口
提供文档入库和问答查询的命令行接口
"""

import sys
import argparse
from pathlib import Path
from argparse import Namespace
from typing import Optional, Dict, Any, List, Sequence

from config import Config, logger, clear_vectorstore
from ingest import ingest_document, batch_ingest
from retrieve_qa import ask_question


def cmd_ingest(args: Namespace) -> None:
    """
    处理文档入库命令
    
    Args:
        args: 命令行参数
    """
    pdf_path = args.pdf_path
    
    # 验证配置
    if not Config.validate():
        logger.error("❌ 配置验证失败，请检查 .env 文件")
        sys.exit(1)
    
    # 执行入库
    result = ingest_document(pdf_path)
    
    if result["success"]:
        print("\n" + "="*60)
        print("✅ 文档入库成功！")
        print("="*60)
        print(f"📄 文档路径: {result['pdf_path']}")
        print(f"📊 文档块数量: {result['chunks_count']}")
        print(f"🏢 提取公司: {', '.join(result['companies'])}")
        print(f"🏭 所属行业: {result['industry']}")
        print(f"💾 数据库总文档数: {result['total_docs_in_db']}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 文档入库失败")
        print("="*60)
        print(f"错误信息: {result.get('error', '未知错误')}")
        print("="*60)
        sys.exit(1)


def cmd_batch_ingest(args: Namespace) -> None:
    """
    处理批量入库命令
    
    Args:
        args: 命令行参数
    """
    directory = args.directory
    
    # 验证配置
    if not Config.validate():
        logger.error("❌ 配置验证失败，请检查 .env 文件")
        sys.exit(1)
    
    # 执行批量入库
    results = batch_ingest(directory)
    
    if not results:
        print("\n" + "="*60)
        print("⚠️ 未找到任何 PDF 文件")
        print("="*60)
        sys.exit(1)
    
    # 统计结果
    success_count = sum(1 for r in results if r.get("success", False))
    fail_count = len(results) - success_count
    
    print("\n" + "="*60)
    print(f"批量入库完成！")
    print("="*60)
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📊 总计: {len(results)}")
    print("="*60)
    
    # 显示失败详情
    if fail_count > 0:
        print("\n失败文件列表：")
        for result in results:
            if not result.get("success", False):
                print(f"  - {result.get('pdf_path', '未知')}: {result.get('error', '未知错误')}")


def cmd_ask(args: Namespace) -> None:
    """
    处理问答命令
    
    Args:
        args: 命令行参数
    """
    question = args.question
    
    # 验证配置
    if not Config.validate():
        logger.error("❌ 配置验证失败，请检查 .env 文件")
        sys.exit(1)
    
    # 执行问答
    result = ask_question(question)
    
    if result["success"]:
        print("\n" + "="*60)
        print("🤖 回答")
        print("="*60)
        print(result["answer"])
        print("\n" + "="*60)
        print(f"📚 引用文档数: {result['docs_count']}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 问答失败")
        print("="*60)
        print(f"错误信息: {result.get('error', '未知错误')}")
        print("="*60)
        sys.exit(1)


def cmd_status(args: Namespace) -> None:
    """
    显示系统状态
    
    Args:
        args: 命令行参数
    """
    print("\n" + "="*60)
    print("📊 RAG 知识库系统状态")
    print("="*60)
    
    # 配置信息
    print(f"\n🔧 配置信息:")
    print(f"  - LLM 模型: {Config.LLM_MODEL_NAME}")
    print(f"  - Embedding 模型: {Config.EMBEDDING_MODEL_NAME}")
    print(f"  - API Base URL: {Config.OPENAI_BASE_URL}")
    print(f"  - 向量数据库路径: {Config.CHROMA_PERSIST_DIR}")
    
    # 检查向量数据库
    try:
        from config import get_vectorstore
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        total_docs = collection.count()
        
        print(f"\n💾 向量数据库:")
        print(f"  - 文档总数: {total_docs}")
        
        # 获取一些元数据统计
        if total_docs > 0:
            # 获取所有文档的元数据
            get_result = collection.get(include=["metadatas"])
            all_metadata: Sequence[Any] = get_result.get("metadatas", []) if get_result else []
            
            # 统计公司
            companies_set = set()
            industries_set = set()
            sources_set = set()
            
            for metadata in all_metadata:
                companies = metadata.get("companies", [])
                if isinstance(companies, list):
                    companies_set.update(companies)
                industry = metadata.get("industry", "")
                if industry:
                    industries_set.add(industry)
                source = metadata.get("source", "")
                if source:
                    sources_set.add(source)
            
            print(f"  - 涉及公司数: {len(companies_set)}")
            print(f"  - 涉及行业数: {len(industries_set)}")
            print(f"  - 文档来源数: {len(sources_set)}")
            
            if companies_set:
                print(f"\n🏢 公司列表:")
                for company in sorted(companies_set):
                    print(f"  - {company}")
            
            if industries_set:
                print(f"\n🏭 行业列表:")
                for industry in sorted(industries_set):
                    print(f"  - {industry}")
            
            if sources_set:
                print(f"\n📄 文档列表:")
                for source in sorted(sources_set):
                    print(f"  - {source}")
        
    except Exception as e:
        print(f"\n⚠️ 无法获取向量数据库信息: {e}")
    
    print("\n" + "="*60)


def cmd_clear(args: Namespace) -> None:
    """
    处理清空数据库命令
    
    Args:
        args: 命令行参数
    """
    # 验证配置
    if not Config.validate():
        logger.error("❌ 配置验证失败，请检查 .env 文件")
        sys.exit(1)
    
    # 确认操作
    print("\n" + "="*60)
    print("⚠️ 警告：此操作将清空向量数据库中的所有文档！")
    print("="*60)
    print("此操作不可恢复，请确认是否继续？")
    print("输入 'yes' 确认，其他任何输入将取消操作")
    
    confirm = input("\n请确认: ").strip()
    
    if confirm.lower() != 'yes':
        print("\n❌ 操作已取消")
        sys.exit(0)
    
    # 执行清空
    success = clear_vectorstore()
    
    if success:
        print("\n" + "="*60)
        print("✅ 向量数据库已清空")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 清空向量数据库失败")
        print("="*60)
        sys.exit(1)


def main() -> None:
    """
    主函数：解析命令行参数并执行相应命令
    """
    parser = argparse.ArgumentParser(
        description="RAG 知识库系统 - 企业级元数据增强检索问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 入库单个 PDF 文档
  python main.py ingest data/raw/report.pdf
  
  # 批量入库目录下的所有 PDF
  python main.py batch-ingest data/raw
  
  # 提问
  python main.py ask "腾讯在人工智能领域有哪些布局？"
  
  # 查看系统状态
  python main.py status
  
  # 清空向量数据库
  python main.py clear
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ingest 命令
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="将 PDF 文档入库到知识库"
    )
    ingest_parser.add_argument(
        "pdf_path",
        type=str,
        help="PDF 文件路径"
    )
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # batch-ingest 命令
    batch_parser = subparsers.add_parser(
        "batch-ingest",
        help="批量入库目录下的所有 PDF 文档"
    )
    batch_parser.add_argument(
        "directory",
        type=str,
        help="PDF 文件所在目录"
    )
    batch_parser.set_defaults(func=cmd_batch_ingest)
    
    # ask 命令
    ask_parser = subparsers.add_parser(
        "ask",
        help="向知识库提问"
    )
    ask_parser.add_argument(
        "question",
        type=str,
        help="您的问题"
    )
    ask_parser.set_defaults(func=cmd_ask)
    
    # status 命令
    status_parser = subparsers.add_parser(
        "status",
        help="显示系统状态"
    )
    status_parser.set_defaults(func=cmd_status)
    
    # clear 命令
    clear_parser = subparsers.add_parser(
        "clear",
        help="清空向量数据库中的所有文档"
    )
    clear_parser.set_defaults(func=cmd_clear)
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有提供命令，显示帮助信息
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 执行对应的命令
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️ 操作已取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 执行命令时发生错误: {e}")
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()