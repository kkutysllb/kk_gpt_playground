#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-27 17:41
# @Desc   : 构建GPT模型接口
# --------------------------------------------------------
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")


class kk_GPT:
    def __init__(self, api_key=API_KEY, base_url=BASE_URL):
        """
        初始化GPT模型接口
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def get_complations(self,
                        messages,
                        model,
                        max_tokens=200,
                        temperature=0.0,
                        stream=False,
                        ):
        """
        获取GPT模型返回的文本

        Args:
            messages (list): 对话内容
            model (str): 模型名称
            max_tokes (int, optional): 返回的最大token数. Defaults to 200.
            temperature (float, optional): 温度. Defaults to 0.0.
            stream (bool, optional): 是否流式返回. Defaults to False.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            logger.error("messages must be a string or a list")
            return ValueError("messages must be a string or a list")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            if stream:
                # 流失输出
                return response
            # 非流式输出
            logger.info(f"获取对话响应成功: {response.choices[0].message.content}")
            logger.info(f"总token数: {response.usage.total_tokens}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"获取对话响应失败: {e}")
            return e
        
    def get_embeddings(self, input):
        """
        获取输入文本的向量表示

        Args:
            input (str): 输入文本
        """
        try:
            embeddings = self.client.embeddings.create(input=input,
                                                     model="text-embedding-3-small",
                                                     dimensions=1536,
                                                     encoding_format="base64",
                                                     )
            return embeddings
        except Exception as e:
            logger.error(f"获取向量表示失败: {e}")
            return e


if __name__ == "__main__":
    gpt = kk_GPT()
    print(gpt.get_complations("你好，请介绍一下你自己", "gpt-4o"))
    
