#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-10-14 21:15
# @Desc   : 大模型调用
# --------------------------------------------------------
"""

from openai import OpenAI
from loguru import logger
from config import LOCAL_API_KEY, LOCAL_API_BASE, LLM_MODELS, OPENAI_API_KEY, OPENAI_API_BASE, ZHIPUAI_API_BASE, ZHIPUAI_API_KEY

def create_chat_response(message, model, temperature, max_tokens, frequency_penalty, presence_penalty, stream_value):
    try:
        if model in LLM_MODELS:
           if model == "qwen2.5-32b-agi":
               api_key = LOCAL_API_KEY
               base_url = LOCAL_API_BASE
           elif model == "gpt-4o-mini":
               api_key = OPENAI_API_KEY
               base_url = OPENAI_API_BASE
           elif model == "glm-4-plus":
               api_key = ZHIPUAI_API_KEY
               base_url = ZHIPUAI_API_BASE
        else:
            return "不支持的模型"
        
        # 模型客户端实例化
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream_value
        )
        # 处理流式响应
        if stream_value:
            return response
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"调用大模型失败: {e}")
        return f"调用大模型失败：{str(e)}"



if __name__ == "__main__":
    message = [{"role": "user", "content": "你好，我来测试下你"}]
    response = create_chat_response(message)
    print(response)