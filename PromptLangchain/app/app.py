#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-26 21:57
# @Desc   : app
# --------------------------------------------------------
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import gradio as gr
import requests
import json
import logging
from typing import List
from basic.config import (
    PORT, MODEL_NAME, TEMPERATURE, MAX_TOKENS, 
    FREQUENCY_PENALTY, PRESENCE_PENALTY, LLM_MODELS, url, headers,
    Messages, ChatCompletionRequest
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def chat_request(
    message: str,
    history: List[List[str]],
    stream: bool
) -> str:
    """发送聊天请求到FastAPI服务"""
    
    # 构建请求
    messages = [Messages(role="user", content=message)]
    request = ChatCompletionRequest(
        messages=messages,
        stream=stream
    )
    # 第一次请求
    data = {
        "messages": [{"role": "user", "content": message}],
        "stream": stream,
        "userId":"123",
        "conversationId":"123"
    }
    
    try:
        if stream:
            # 流式输出
            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
            partial_response = ""
            
            # 先添加用户消息到历史记录
            history.append([message, ""])  # 添加用户消息和空的助手回复
            
            for line in response.iter_lines():
                if line:
                    json_str = line.decode('utf-8').strip("data: ")
                    if not json_str:
                        continue
                    if json_str.startswith("{") and json_str.endswith("}"):
                        try:
                            json_data = json.loads(json_str)
                            if json_data['choices'][0]['finish_reason'] == 'stop':
                                break
                            delta = json_data['choices'][0]['delta'].get('content', '')
                            partial_response += delta
                            # 更新最后一条助手回复
                            history[-1][1] = partial_response
                            yield history
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解码失败: {e}")
        else:
            # 非流式输出
            response = requests.post(url, headers=headers, data=json.dumps(data))
            content = response.json()['choices'][0]['message']['content']
            if isinstance(content, str) and content.startswith("content='") and content.endswith("'"):
                content = content[8:-1]
            # 添加完整的对话到历史记录
            history.append([message, content])
            yield history
    
    except Exception as e:
        logger.error(f"请求失败: {e}")
        raise f"请求失败: {str(e)}"
    
    

def create_chat_interface():
    """创建聊天界面"""
    with gr.Blocks(title="智能客服系统") as demo:
        gr.HTML("""
            <h1 style="text-align: center; margin-bottom: 1rem">
                智能客服系统
            </h1>
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=600,
                    show_label=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="请输入您的问题...",
                        show_label=False,
                        container=False,
                    )
                    submit_btn = gr.Button("发送")
            
            with gr.Column(scale=1):
                model = gr.Dropdown(
                    choices=LLM_MODELS,
                    value=LLM_MODELS[0],
                    label="选择模型"
                )
                stream = gr.Checkbox(
                    value=True,
                    label="启用流式输出"
                )
                clear_btn = gr.Button("清空对话")
            submit_btn.click(
                fn=chat_request,
                inputs=[msg, chatbot, stream],
                outputs=[chatbot]
            )

        # 清空对话
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        
    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)