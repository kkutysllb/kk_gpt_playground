#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-27 21:35
# @Desc   : kk_GPT 翻译器
# --------------------------------------------------------
"""
import os
import sys
import openai
from loguru import logger
from dotenv import load_dotenv

from file_processor_helper import FileProcessorHelper
from config import API_KEY, BASE_URL, EMBEDDINGS_MODEL

load_dotenv()

class kk_GPT:
    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
    def get_completions(
        self,
        messages,
        model,
        max_tokens=4096,
        temperature=0.5,
        stream=True
    ):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            return f"messages 必须是字符串或消息列表，当前类型为 {type(messages)}"
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
        
        if stream:
            return response
        else:
            logger.success(f"非流式输出 | : total_tokens={response.usage.total_tokens}"
                           f"= prompt_tokens: {response.usage.prompt_tokens}"
                           f"+ completion_tokens: {response.usage.completion_tokens}")
            return response.choices[0].message.content
        
        
    def get_embbeddings(self, input):
        response = self.client.embeddings.create(input=input, model=EMBEDDINGS_MODEL)
        embeddings = [data.embedding for data in response.data]
        return embeddings
    


if __name__ == "__main__":
    # 测试
    kk_gpt = kk_GPT()
    
    prompt = "你好"
    bot_response = kk_gpt.get_completions(prompt, "glm-4-plus")
    completion = ""
    for character in bot_response:
        character_content = character.choices[0].delta.content
        if character_content is not None:
            completion += character_content
            logger.success(f"流式输出 | bot_response: {completion}")
        else:
            prompt_tokens = FileProcessorHelper.tiktoken_len(prompt)
            completion_tokens = FileProcessorHelper.tiktoken_len(completion)
            total_tokens = prompt_tokens + completion_tokens
            logger.success(f"流式输出 | prompt_tokens: {prompt_tokens} + completion_tokens: {completion_tokens} = total_tokens: {total_tokens}")
            logger.success(f"流式输出 | prompt_tokens: {prompt_tokens} + completion_tokens: {completion_tokens} = total_tokens: {total_tokens}")
    embeddings = kk_gpt.get_embbeddings(prompt)
    logger.success(f"embeddings: {embeddings}")
