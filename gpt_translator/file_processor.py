#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-27 21:46
# @Desc   : 文件处理
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import hashlib
from typing import Dict, List, Any, Union


class FileProcessor():
    # 定义允许处理的文件后缀
    ALLOWED_EXTENSIONS = [".pdf", ".txt"]

    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_file_extension(self) -> str:
        """
        获取文件后缀
        :return:
        """
        _, file_extension = os.path.splitext(self.file_path)
        return file_extension.lower()

    def is_alowed_file(self):
        """
        判断文件是否允许处理
        :return:
        """
        file_extension = self.get_file_extension()
        return file_extension in self.ALLOWED_EXTENSIONS

    def get_file_name(self):
        """
        获取文件名
        :return:
        """
        file_name = os.path.basename(self.file_path)
        return file_name

    def get_file_md5(self):
        """
        获取文件md5
        :return:
        """
        file_bytes = self.get_file_bytes(self.file_path)
        file_md5 = self.calculate_md5(file_bytes)
        return file_md5

    @staticmethod
    def get_file_bytes(file_path: str):
        """
        获取文件字节
        :return:
        """
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return file_bytes

    @staticmethod
    def calculate_md5(input_data: Union[str, bytes]) -> str:
        """
        计算文件的md5值
        :param file_path:
        :return:
        """
        md5 = hashlib.md5()
        # 判断输入是字符串还是字节流
        if isinstance(input_data, str):
            md5.update(input_data.encode('utf-8'))
        elif isinstance(input_data, bytes):
            md5.update(input_data)
        else:
            raise ValueError("Invalid input type. Input must be a string or bytes.")
        return md5.hexdigest()


if __name__ == '__main__':
    # 测试
    file_path = os.path.join(root_dir, "data", "LangChain整体项目介绍与核心模块Model IO详解.pdf")
    file_processor = FileProcessor(file_path)
    print(file_processor.get_file_extension())
    print(file_processor.is_alowed_file())
