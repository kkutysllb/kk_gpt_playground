#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-27 22:17
# @Desc   : web ui with gradio
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import gradio as gr
import pandas as pd

from config import MODELS, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, MODEL_TO_MAX_TOKENS, API_KEY, BASE_URL
from kk_GPT import kk_GPT
from file_processor_helper import FileProcessorHelper
from utils import build_chat_document_prompt, upload_files
from loguru import logger

logger.remove() # 删去import logger之后自动产生的handler，不删除的话会出现重复输出的现象
logger.add(sys.stderr, level="DEBUG")



def fn_update_max_tokens(model, origin_set_tokens):
    """
    模型有改动时，对应的 max_tokens_slider 滑块组件的最大值随之改动。
    """
    new_max_tokens = MODEL_TO_MAX_TOKENS[model]
    new_max_tokens = new_max_tokens if new_max_tokens else origin_set_tokens
    
    new_set_tokens = origin_set_tokens if origin_set_tokens <= new_max_tokens else DEFAULT_MAX_TOKENS
    
    new_max_tokens_component = gr.Slider(
        label="max tokens",
        value=new_set_tokens,
        minimum=0,
        maximum=new_max_tokens,
        step=1,
        interactive=True
    )
    
    return new_max_tokens_component


def fn_prehandle_user_input(user_input, chat_history):
    """
    用户输入框处于焦点状态时按 Enter 键时，将触发此侦听器。
    """
    logger.info(f"用户输入: {user_input}, 聊天历史: {chat_history}")
    
    chat_history = [] if not chat_history else chat_history
    
    # 检查输入
    if not user_input:
        gr.Warning("请输入您的问题")
        return chat_history
    
    # 将用户输入添加到聊天历史中
    chat_history.append((user_input, None))
    
    return chat_history


def fn_chat(
    chat_mode,
    uploaded_file_path_df,
    user_input,
    chat_history,
    model,
    max_tokens,
    temperature,
    stream,
    top_n_number
):
    """
    聊天功能
    """
    # 如果用户输入为空，则返回当前的聊天记录
    if not user_input:
        return chat_history
    
    # 获取已上传文件的路径列表
    uploaded_file_path_list = uploaded_file_path_df['已上传的文件'].values.tolist()
    
    # 打印日志，记录输入参数信息
    logger.info(f"\n"
                f"问答模式: {chat_mode} \n"
                f"文件路径: {uploaded_file_path_list} {type(uploaded_file_path_list)} \n"
                f"用户输入: {user_input} \n"
                f"历史记录: {chat_history} \n"
                f"使用模型: {model} {type(model)}\n"
                f"要生成的最大token数: {max_tokens} {type(max_tokens)}\n"
                f"温度: {temperature} {type(temperature)}\n"
                f"是否流式输出: {stream} {type(stream)}\n"
                f"top_n: {top_n_number} {type(top_n_number)}")
    
    # 构建messages参数
    messages = []
    if chat_mode == "普通问答":
        messages = [
            {"role": "system", "content": "你是一个聊天机器人，请根据用户的问题进行回答。"},
            {"role": "user", "content": user_input}
        ]
        if len(chat_history) > 1:
            messages = []
            for chat in chat_history:
                if chat[0] is not None:
                    messages.append({"role": "user", "content": chat[0]})
                if chat[1] is not None:
                    messages.append({"role": "assistant", "content": chat[1]})
    else:
        # 文档问答
        # 检查是否上传了文件
        # uploaded_file_path_list 是一个非空列表，包含上传文件的路径
        # 如果 uploaded_file_path_list 不是列表，或者是空列表，或者包含空字符串，则抛出错误
        if not isinstance(uploaded_file_path_list, list) or not uploaded_file_path_list or '' in uploaded_file_path_list:
            gr.Warning("请上传文件")
            return chat_history
        
        user_prompt = build_chat_document_prompt(
            uploaded_file_path_list,
            user_input,
            chat_history,
            top_n_number
        )
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        else:
            logger.error("生成 user_prompt 失败")
            messages = []
    
    # 检查 messages 参数
    if not messages:
        logger.error(f"messages为空列表")
        gr.Warning("服务器错误")
        return chat_history
    else:
        # 打印 messages 参数
        logger.info(f"messages: {messages}")
        
        # messages有值，生成回复
        gpt = kk_GPT()
        bot_response = gpt.get_completions(
            messages, model, max_tokens, temperature, stream)
        
        if stream:
            # 流式输出
            chat_history[-1][1] = ""
            for character in bot_response:
                character_content = character.choices[0].delta.content
                if character_content is not None:
                    chat_history[-1][1] += character_content
                    yield chat_history
                else:
                    logger.success(f"流式输出 | bot_response: {chat_history[-1][1]}")
                    # 估算流式输出的 token 用量
                    # prompt: messages 里的所有字符拼在一起
                    prompt = messages # messages 类型可能是 str，也可能是 list
                    if isinstance(messages, list):
                        prompt = ""
                        for message in messages:
                            prompt += message['content'] + "\n"
                    logger.trace(f"prompt: {prompt}")
                    # prompt 的 token 数量
                    prompt_tokens = FileProcessorHelper.tiktoken_len(prompt)
                    # completion 的 token 数量
                    completion_tokens = FileProcessorHelper.tiktoken_len(chat_history[-1][1])
                    # 总 token 数量
                    total_tokens = prompt_tokens + completion_tokens
                    logger.success(f"流式输出 | total_tokens: {total_tokens} "
                                   f"= prompt_tokens:{prompt_tokens} + completion_tokens: {completion_tokens}")
        else:
            # 非流式输出
            chat_history[-1][1] = bot_response
            logger.success(f"非流式输出 | bot_response: {chat_history[-1][1]}")
            yield chat_history
            
def fn_upload_files(unuploaded_file_paths):
    """
    上传文件
    """
    # 初始化上传成功的文件列表
    uploaded_file_paths = []
    
    # 循环处理待上传的文件
    for file_path in unuploaded_file_paths:
        # 调用上传文件函数
        result = upload_files(str(file_path))
        # 处理函数结果
        if result.get('code') == 200:
            # 上传成功
            gr.Info("文件上传成功！")
            uploaded_file_paths.append(
                result.get('data').get('uploaded_file_path'))
        else:
            # 上传失败
            raise gr.Error("文件上传失败！")    
        
    return pd.DataFrame({'已上传的文件': uploaded_file_paths})



with gr.Blocks() as demo:
    gr.Markdown("# <center> kk_GPT 翻译器</center>")
    # 定义一个行布局，内含两个等高的列布局
    with gr.Row(equal_height=True):
        # 左侧列布局，对话框
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="聊天机器人")
            user_input = gr.Textbox(label="用户输入框", placeholder="这篇文章讲了什么")
            with gr.Row():
                submit_btn = gr.Button("提交")
                clear_btn = gr.Button("清除")
        # 右侧列布局，文件上传, 模型参数
        with gr.Column(scale=1):
            # 创建一个选项卡，用于设置问答选项
            with gr.Tab(label="问答选项"):
                chat_mode_radio = gr.Radio(
                    label="问答模式",
                    choices=["普通问答", "文档问答"],
                    value="文档问答",
                    interactive=True
                )
                file_path_files = gr.Files(
                    label="文件上传",
                    file_count="multiple",
                    file_types=[".txt", ".pdf"],
                    type="filepath",  # 组件要返回值的类型
                )
                file_path_dataframe = gr.Dataframe(
                    value=pd.DataFrame({'已上传的文件': []})
                )
                top_n_number = gr.Number(
                    label="top n",
                    value=20,
                    interactive=True
                )
            # 创建一个选项卡，用于调整参数
            with gr.Tab(label="模型参数"):
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        label="模型选择",
                        choices=MODELS,
                        value=DEFAULT_MODEL,
                        multiselect=False,
                        interactive=True
                    )
                    max_tokens_slider = gr.Slider(
                        label="max tokens",
                        value=4096,
                        minimum=0,
                        maximum=8192,
                        step=1,
                        interactive=True
                    )
                    temperature_slider = gr.Slider(
                        label="temperature",
                        value=0.7,
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        interactive=True
                    )
                    stream_radio = gr.Radio(
                        label="流式输出",
                        choices=[True, False],
                        value=True,
                        interactive=True
                    )

    # 模型有改动时，对应的 max_tokens_slider 滑块组件的最大值随之改动。
    model_dropdown.change(
        fn=fn_update_max_tokens,
        inputs=[model_dropdown, max_tokens_slider],
        outputs=max_tokens_slider
    )
    
    # 当用户在文本框处于焦点状态时按 Enter 键时，将触发此侦听器。
    user_input.submit(
        fn=fn_prehandle_user_input,
        inputs=[user_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=fn_chat,
        inputs=[
            chat_mode_radio,
            file_path_dataframe,
            user_input,
            chatbot,
            model_dropdown,
            max_tokens_slider,
            temperature_slider,
            stream_radio,
            top_n_number
        ],
        outputs=[chatbot]
    )
    
    # 单击按钮时触发。
    submit_btn.click(
        fn=fn_prehandle_user_input,
        inputs=[user_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=fn_chat,
        inputs=[
            chat_mode_radio,
            file_path_dataframe,
            user_input,
            chatbot,
            model_dropdown,
            max_tokens_slider,
            temperature_slider,
            stream_radio,
            top_n_number
        ],
        outputs=[chatbot]
    )
    
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    # 上传文件时触发。
    file_path_files.upload(
        fn=fn_upload_files,
        inputs=[file_path_files],
        outputs=[file_path_dataframe],  # 展示已上传的文件
        show_progress=True,  # 如果为 True，则在挂起时显示进度动画
    )
    
if __name__ == "__main__":
    demo.queue().launch()

