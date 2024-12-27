#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-10-14 21:15
# @Desc   : 工具脚本
# --------------------------------------------------------
"""

from gpt_chat_handler import create_chat_response
from loguru import logger


def llm_reply(chat_history,
              user_input,
              model,
              temperature,
              max_tokens,
              frequency_penalty,
              presence_penalty,
              stream_radio
              ):
    
    stream_value = True if stream_radio == "流式输出" else False
    
    # 用户消息在前端对话展示
    chat_history.append([user_input, None])
    
    # 初始化message为空列表
    messages = []
    
    # 如果对话历史长度超过1，则遍历历史记录构建messages
    if len(chat_history) > 1:
        for user_msg, assistant_msg in chat_history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    else:
        # 如果没有对话历史记录，则直接使用用户输入
        messages.append({"role": "user", "content": user_input})
    
    # 调用大模型
    gpt_response = create_chat_response(messages,
                                        model,
                                        temperature,
                                        max_tokens,
                                        frequency_penalty,
                                        presence_penalty,
                                        stream_value)
    if stream_value:
        # 流式输出
        chat_history[-1][1] = ""
        for chunk in gpt_response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                chat_history[-1][1] += chunk_content
                yield chat_history
    else:
        # 非流式输出
        chat_history[-1][1] = gpt_response
        logger.info(f"\n对话历史：{chat_history}")
        yield chat_history
    
    # 后台日志打印
    logger.info(f"\n用户输入: {user_input}, "
                f"\n模型：{model}, "
                f"\n温度：{temperature}, "
                f"\n最大输入token：{max_tokens}, "
                f"\n惩罚频率：{frequency_penalty}, "
                f"\n惩罚值：{presence_penalty},"
                f"\n输出模式：{stream_radio},"
                f"\n模型响应：{gpt_response}")
    
    # chatbot接收的是列表形式的输入
    return [[user_input, gpt_response]]