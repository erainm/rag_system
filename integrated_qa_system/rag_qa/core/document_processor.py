# Created by erainm on 2025/10/26 15:34.
# IDE：PyCharm 
# @Project: rag_system
# @File：document_processor
# @Description: 文档处理

"""
需求：从指定文件夹加载多种类型文件并添加元数据
思路：
1. 遍历目录下的文件
2. 过滤文件类型，并加载文件
    2.1 基于文件类型构造加载器类型。如果是txt，需要指定编码为utf-8。
    2.2 加载对应的文件并转为Document
3. 给每个文档添加元数据：学科、路径、时间戳
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter
from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from datetime import datetime
import sys
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from rag_qa.edu_document_loaders.edu_docloader import OCRDOCLoader
from rag_qa.edu_document_loaders.edu_imgloader import OCRIMGLoader
from rag_qa.edu_document_loaders.edu_pdfloader import OCRPDFLoader
from rag_qa.edu_document_loaders.edu_pptloader import OCRPPTLoader
from base.logger import logger
from base.config import conf

# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path):
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 获取支持的文件扩展名集合
    supported_extensions = document_loaders.keys()
    # print(f'supported_extensions--》{supported_extensions}')
    # 从目录名提取学科类别（如 "ai_data" -> "ai"）
    # print(f'1---》{os.path.basename(directory_path)}')
    source = os.path.basename(directory_path).replace("_data", "")
    # print(f'source-->{source}')
    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory_path):
        # print(f'root---》{root}')
        # print(f'files---》{files}')
        # 遍历当前目录下的所有文件
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # print(f'file_path--》{file_path}')
            # print(os.path.splitext(file_path))
            # 获取文件扩展名并转换为小写
            file_extension = os.path.splitext(file_path)[1].lower()
            # print(f'file_extension--》{file_extension}')
            # 检查文件类型是否在支持的扩展名列表中
            if file_extension in supported_extensions:
                # 使用 try-except 捕获加载过程中的异常
                try:
                    # 根据文件扩展名获取对应的加载器类
                    loader_class = document_loaders[file_extension]
                    # 实例化加载器对象，传入文件路径
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 调用加载器加载文档内容，返回文档列表
                    loaded_docs = loader.load()
                    # print(f'loaded_docs--》{loaded_docs}')
                    # print(f'loaded_docs--》{len(loaded_docs)}')
                    for doc in loaded_docs:
                        # 为文档添加学科类别元数据
                        doc.metadata["source"] = source
                        # 为文档添加文件路径元数据
                        doc.metadata["file_path"] = file_path
                        # 为文档添加当前时间戳元数据
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    # print(f'loaded_docs111--》{loaded_docs}')
                    documents.extend(loaded_docs)
                    # 记录成功加载文件的日志
                    logger.info(f"成功加载文件: {file_path}")
                except Exception as e:
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            # 如果文件类型不在支持列表中
            else:
                # 记录警告日志，提示不支持的文件类型
                logger.warning(f"不支持的文件类型: {file_path}")
    # 返回加载的所有文档列表
    return documents

# 定义函数，处理文档并进行分层切分，返回子块结果
def process_documents(directory_path,
                      parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    # 从指定目录加载所有文档
    documents = load_documents_from_directory(directory_path)
    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化父块和子块分词器（通用）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用分词器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 初始化空列表，用于存储所有子块
    child_chunks = []
    #遍历每个原始文档，带上索引 i
    for i, doc in enumerate(documents):
        # print(f'doc--》{doc}')
        # print(doc.metadata.get("file_path", ''))
        # # 获取文件的扩展名
        # print(os.path.splitext(doc.metadata.get("file_path", '')))
        file_extension = os.path.splitext(doc.metadata.get("file_path", ''))[1].lower()
        # print(f'file_extension--》{file_extension}')
        # 选择分词器
        is_markdown = (file_extension == '.md')
        # print(f'is_markdown--》{is_markdown}')
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        # print(f'parent_splitter_to_use-->{parent_splitter_to_use}')
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}")
        # 使用父块切分器将文档切分为父块
        parent_docs = parent_splitter_to_use.split_documents([doc])
        # print(f'parent_docs--》{parent_docs}')
        # print(f'parent_docs--》{len(parent_docs)}')
        # 遍历每个父块，带上索引 j,切分子块
        for j, parent_doc in enumerate(parent_docs):
            # 为父块生成唯一 ID，格式为 "doc_i_parent_j"
            parent_id = f"doc_{i}_parent_{j}"
            # # 将父块 ID 添加到元数据
            # parent_doc.metadata["parent_id"] = parent_id
            # # 将父块内容存储到元数据
            # parent_doc.metadata["parent_content"] = parent_doc.page_content

            # 使用子块分词器将父块切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # print(f'sub_chunks--》{sub_chunks}')
            # print(f'sub_chunks--》{len(sub_chunks)}')
            # 遍历每个子块，为子块主要添加对应的父块文档
            for k, sub_chunk in enumerate(sub_chunks):
                # print(f'原始的sub_chunk--》{sub_chunk}')
                # 为子块添加父块的ID
                sub_chunk.metadata["parent_id"] = parent_id
                # 为子块添加对应的父块文档（元数据）
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # 为子块生成一个唯一的ID,格式为 "parent_id_child_k"
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                # print(f'修改后的sub_chunk--》{sub_chunk}')
                # 将子块添加到子块列表中
                child_chunks.append(sub_chunk)

    # 记录子块总数日志
    logger.info(f"子块数量: {len(child_chunks)}")
    # 返回所有子块列表
    return child_chunks
if __name__ == '__main__':
    directory_path = '/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/data/ai_data'
    # documents = load_documents_from_directory(directory_path)
    # print(documents)
    child_chunks = process_documents(directory_path)
    print(f'child_chunks--》{child_chunks[0]}')