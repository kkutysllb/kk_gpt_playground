#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-10-14 21:15
# @Desc   : Gradio构建的Web UI
# --------------------------------------------------------
"""
import gradio as gr
from utils import llm_reply
from config import LLM_MODELS


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            # 左侧对话栏
            with gr.Column():
                chatbot = gr.Chatbot(label="边缘云智能聊天机器人")
                user_input = gr.Textbox(label="输入框", placeholder="你好，请在这里请输入你的问题")
                with gr.Row():
                    user_submit = gr.Button("提交")
                    user_clear = gr.Button("清除")
            # 右侧参数栏
            with gr.Column():
                model_dropdown=gr.Dropdown(
                    choices=LLM_MODELS,
                    value=LLM_MODELS[0],
                    label="模型选择",
                    interactive=True
                )
                temperature_slider = gr.Slider(label="Temperature",
                          minimum=0,
                          maximum=1,
                          value=0.5,
                          )
                max_tokens_slider = gr.Slider(label="Max Tokens",
                          minimum=1024,
                          maximum=8192,
                          value=4096,
                          )
                frequency_penalty_slider = gr.Slider(label="Frequency Penalty",
                          minimum=-2,
                          maximum=2,
                          value=0, 
                          )
                presence_penalty_slider = gr.Slider(label="Presence Penalty",
                          minimum=-2,
                          maximum=2,
                          value=0,
                          )
                # 输出模式选择
                stream_radio = gr.Radio(
                    choices=["流式输出", "非流式输出"],
                    value="流式输出",
                    label="输出模式",
                )
                
            # 用户点击事件
            user_submit.click(
                fn=llm_reply,
                inputs=[chatbot,
                        user_input, 
                        model_dropdown, 
                        temperature_slider, 
                        max_tokens_slider, 
                        frequency_penalty_slider, 
                        presence_penalty_slider, 
                        stream_radio
                        ],
                outputs=[chatbot]
            )
    return demo



if __name__ == "__main__":
    main().launch()
