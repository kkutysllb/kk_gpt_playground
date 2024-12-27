#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-10-14 21:15
# @Desc   : web ui app with gradio
# --------------------------------------------------------
"""
import gradio as gr
from config import MODELS, DEFAULT_MODEL, MODEL_TO_MAX_TOKENS
from kk_GPT import kk_GPT
import logging
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("LOCAL_API_KEY")
BASE_URL = os.getenv("LOCAL_API_BASE")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 实例化kk_GPT, 本地部署的qwen2.5-32b-agi模型
kk_gpt = kk_GPT(api_key=API_KEY, base_url=BASE_URL)



def fn_update_max_tokens(model, original_max_tokens):
    """
    模型选择改变时，更新最大token数
    """
    # 获取模型对应的新最大token数，如果没有设置则使用传入的最大token数
    new_max_tokens = MODEL_TO_MAX_TOKENS.get(model)
    new_max_tokens = new_max_tokens if new_max_tokens else original_max_tokens
    
    # 如果原始设置的令牌数超过了新的最大令牌数，将其调整为默认值500
    new_set_tokens = original_max_tokens if original_max_tokens <= new_max_tokens else 500
    
    # 创建新的最大令牌数slider
    new_max_tokens_component = gr.Slider(
        label="Max Tokens",
        minimum=0,
        maximum=new_max_tokens,
        value=new_set_tokens,
        step=1,
        interactive=True
    )
    return new_max_tokens_component


def fn_prehandle_user_input(user_input, chat_history):
    """
    用户输入框内容预处理
    """
    if not user_input:
        gr.Warning("请输入内容")
        logger.warning("请输入内容")
        return chat_history
    
    #用户消息在前端对话框展示
    chat_history.append([user_input, None])
    logger.info(f"用户输入：{user_input}, \n"
                f"聊天历史：{chat_history}"
                )
    
    return chat_history



def fn_predict(user_input, chat_history, model, temperature, max_tokens, stream):
    """
    预测用户输入
    """
    if not user_input:
        return chat_history
    
    logger.info(f"用户输入：{user_input}, \n"
                f"聊天历史：{chat_history}, \n"
                f"模型：{model}, \n"
                f"温度：{temperature}, \n"
                f"最大令牌数：{max_tokens}, \n"
                f"流式输出：{stream}"
                )
    
    # 构建messages参数
    messages= [
        {"role": "system", "content": "你是一个知识助手，回答用户的问题。"},
        {"role": "user", "content": user_input}
    ]
    
    if len(chat_history) > 1:
        messages = []
        for chat in chat_history:
            if chat[0] is not None:
                messages.append({"role": "user", "content": chat[0]})
            if chat[1] is not None:
                messages.append({"role": "assistant", "content": chat[1]})
    logger.info(f"构建的messages参数：{messages}")
    
    # 生成回复
    bot_response = kk_gpt.get_complations(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream
    )
    
    if stream:
        # 流式输出
        chat_history[-1][1] = ""
        for character in bot_response:
            character_content = character.choices[0].delta.content
            if character_content is not None:
                chat_history[-1][1] += character_content
                yield chat_history
    else:
        # 非流式输出
        chat_history[-1][1] = bot_response
        yield chat_history


with gr.Blocks() as demo:
    gr.Markdown("# 欢迎使用kk_GPT知识问答系统")
    with gr.Row(equal_height=True):
        # 左侧对话栏
        with gr.Column(scale=4):
            chat_bot = gr.Chatbot(label="小智知识助手")
            user_input = gr.Textbox(label="用户输入框", placeholder="你好，请输入你的问题...")
            with gr.Row():
                user_submit = gr.Button("发送")
                user_clear = gr.Button("清除")
        # 右侧参数栏
        with gr.Column(scale=1):
            # 创建一个包括三个参数输入的选项
            with gr.Tab(label="模型参数"):
                model_dropdown = gr.Dropdown(
                    label="模型选择",
                    choices=MODELS,
                    value=DEFAULT_MODEL,
                    multiselect=False,
                    interactive=True
                )
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    interactive=True
                )
                max_tokens_slider = gr.Slider(
                    label="Max Tokens",
                    minimum=0,
                    maximum=8192,
                    value=4096,
                    step=1,
                    interactive=True
                )
                stream_radio = gr.Radio(
                    label="流式输出",
                    choices=[True, False],
                    value=True,
                    interactive=True
                )
        # 模型选择改变时，更新最大token数
        model_dropdown.change(
            fn=fn_update_max_tokens,
            inputs=[model_dropdown, max_tokens_slider],
            outputs=max_tokens_slider
        )
        # 当用户在文本框处于焦点状态时按Enter，将触发此监听器
        user_input.submit(
            fn = fn_prehandle_user_input,
            inputs=[user_input, chat_bot],
            outputs=[chat_bot]
        )
        # 用户点击发送按钮时，触发此监听器
        user_submit.click(
            fn=fn_prehandle_user_input,
            inputs=[user_input, chat_bot],
            outputs=[chat_bot]
        ).then(
            fn=fn_predict,
            inputs=[
            user_input,
            chat_bot,
            model_dropdown,
            temperature_slider,
            max_tokens_slider,
            stream_radio
            ],
            outputs=[chat_bot]
        )
        
        # 用户点击清除按钮时，触发此监听器
        user_clear.click(
            lambda: None,
            None,
            chat_bot,
            queue=False
        )
        
    

if __name__ == "__main__":
    demo.queue().launch()