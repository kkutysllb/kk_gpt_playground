#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-27 21:46
# @Desc   : 文件处理器
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


import tiktoken
import pdfplumber
from typing import List, Dict, Any
from config import CHUNK_SIZE, CHUNK_OVERLAP, API_KEY, BASE_URL
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter



class FileProcessorHelper:
    def __init__(self,
                 file_path: str,
                 file_name: str = None,
                 file_extension: str = None,
                 file_md5: str = None,
                 ) -> None:
        self.file_path = file_path
        self.file_name = file_name
        self.file_extension = file_extension
        self.file_md5 = file_md5
        
    def get_file_to_docs(self) -> List:
        """
        获取文件到文档
        """
        stratege_mapping = {
            ".txt": self.get_txt_to_docs,
            ".pdf": self.get_pdf_to_docs,
            # "docx": self.get_docx_to_docs,
            # "md": self.get_md_to_docs,
        }
        func = stratege_mapping.get(self.file_extension)
        return func(self.file_path)
    
    def split_docs(self, docs):
        """
        分割文档
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, 
                                                       chunk_overlap=CHUNK_OVERLAP,
                                                       length_function=self.tiktoken_len,
                                                       )
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        docs = text_splitter.create_documents(texts, metadatas)
        return docs
    
    
    @staticmethod
    def get_pdf_to_docs(file_path: str) -> List:
        """
        获取pdf文件到文档
        """
        file_name = os.path.basename(file_path)
        
        docs = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    doc = Document(
                        page_content=page_text,
                        metadata=dict(
                            {
                                "file_name": file_name,
                                "page_number": page.page_number,
                                "total_pages": len(pdf.pages),
                            },
                            **{
                                k: pdf.metadata[k] for k in pdf.metadata if isinstance(pdf.metadata[k], (str, int)) 
                            }
                        )
                    )
                    docs.append(doc)
        return docs
    
    
    @staticmethod
    def get_txt_to_docs(file_path: str) -> List:
        """
        获取txt文件到文档
        """
        file_name = os.path.basename(file_path)
        
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        if not text:
            return []
        return [Document(page_content=text, metadata={"file_name": file_name})]
    
    @staticmethod
    def tiktoken_len(text: str) -> int:
        """
        获取文本长度
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text,
                                 disallowed_special=set()  # 禁用对所有特殊标记的检查
                                 )
        return len(tokens)


if __name__ == "__main__":
    # 测试
    file_path = os.path.join(root_dir, "data", "LangChain整体项目介绍与核心模块Model IO详解.pdf")
    file_name = "LangChain整体项目介绍与核心模块Model IO详解.pdf"
    file_extension = ".pdf"
    file_md5 = "e41ab92c3f938ddb3e82110becbbce3e"
    
    file_processor_helper = FileProcessorHelper(file_path, file_name, file_extension, file_md5)
    docs = file_processor_helper.get_file_to_docs()
    print(docs)
    
    # 测试tokens
    print(file_processor_helper.tiktoken_len("你好，世界！"))
    
    