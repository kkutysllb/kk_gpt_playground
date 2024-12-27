#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-25 15:24
# @Desc   : 主函数，实现一个Fast API接口完成模型的初始化和全生命周期管理
# --------------------------------------------------------
"""
import os
import time
import uuid
import json
import asyncio

from contextlib import asynccontextmanager

# langchain框架
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 部署REST API相关
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from config import MODEL_NAME, TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, PORT, PROMPT_TEMPLATE_SYSTEM, PROMPT_TEMPLATE_USER
from config import logger, get_prompt, get_model_config, format_response, ChatCompletionRequest, ChatComplationResponseChoice, ChatCompletionResponse, Messages


# 在文件开头添加环境变量设置
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 申明全局变量，全局调用
model = None
prompt = None
chain = None


# 定义了一个异步函数lifespan，它接收一个FastAPI应用实例app作为参数。这个函数将管理应用的生命周期，包括启动和关闭时的操作
# 函数在应用启动时执行一些初始化操作，如设置搜索引擎、加载上下文数据、以及初始化问题生成器
# 函数在应用关闭时执行一些清理操作
# @asynccontextmanager 装饰器用于创建一个异步上下文管理器，它允许你在 yield 之前和之后执行特定的代码块，分别表示启动和关闭时的操作
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在执行时启动
    # 申明引用全局变量
    global model, prompt, chain
    # 获得模型基础配置
    api_key, base_url = get_model_config(MODEL_NAME)
    # 根据自己实际情况选择调用model和embedding模型类型
    try:
        logger.info(f"模型初始化开始, 初始化模型基本配置，Prompt模板，Chain...")
        model = ChatOpenAI(api_key=api_key, 
                           base_url=base_url,
                           model=MODEL_NAME,
                           temperature=TEMPERATURE,
                           max_tokens=MAX_TOKENS,
                           frequency_penalty=FREQUENCY_PENALTY,
                           presence_penalty=PRESENCE_PENALTY
                           )
        # 提取prompt模板
        prompt_system = PromptTemplate.from_file(PROMPT_TEMPLATE_SYSTEM)
        prompt_user = PromptTemplate.from_file(PROMPT_TEMPLATE_USER)
        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', prompt_system.template),
                ('user', prompt_user.template)
            ]
        )
        # 定义chain
        chain = prompt | get_prompt | model
        # 初始化完成日志打印
        logger.info(f"模型初始化完成")
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        raise e

    # yield 关键字将控制权交还给FastAPI框架，使应用开始运行
    # 分隔了启动和关闭的逻辑。在yield 之前的代码在应用启动时运行，yield 之后的代码在应用关闭时运行
    yield
    # 在应用关闭时执行一些清理操作
    logger.info(f"模型关闭")
    
app = FastAPI(lifespan=lifespan)

# POST请求接口，与大模型进行知识问答
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    if not model or not prompt or not chain:
        logger.error(f"模型未初始化")
        raise HTTPException(status_code=500, detail="模型未初始化")
    try:
        logger.info(f"收到聊天完成请求: {request}")
        query_prompt = request.messages[-1].content
        logger.info(f"用户的问题是: {query_prompt}")
        # 调用chain进行推理
        result = chain.invoke({"query": query_prompt})
        if not isinstance(result, str):
            result = str(result)
        # 对响应进行格式化
        formatted_response = str(format_response(result))
        logger.info(f"格式化的模型推理结果: {formatted_response}")
        
        # 处理流式响应
        if request.stream:
            # 定义一个异步生成器函数，用于生成流式数据
            async def generate_stream():
                # 为每个流式片段生成一个唯一的chunk_id
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                # 将格式化的响应分割
                lines = formatted_response.split('\n')
                # 遍历每一行，并构建响应片段
                for i, line in enumerate(lines):
                    # 创建一个字典，表示流式响应片段
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + '\n'},
                                "finish_reason": None
                            }
                        ]
                    }
                    # 将片段转换为json格式并生成
                    yield f"{json.dumps(chunk)}\n"
                    # 每次生成数据后，等待0.5秒
                    await asyncio.sleep(0.5)
                # 生成最后一个片段，表示流式响应结束
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                # 将最后一个片段转换为json格式并生成
                yield f"{json.dumps(final_chunk)}\n"
            # 返回fastapi.responses中StreamingResponse对象，流式传输数据
            # media_type设置为text/event-stream以符合SSE(Server-SentEvents) 格式
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        else:
            # 非流式响应
            response = ChatCompletionResponse(
                choices=[
                    ChatComplationResponseChoice(
                        index=0,
                        message=Messages(role="assistant", content=formatted_response),
                        finish_reason="stop"
                    )
                ]
            )
            logger.info(f"发送响应内容: {response}")
            # 返回fastapi.responses中JSONResponse对象
            # model_dump()方法通常用于将Pydantic模型实例的内容转换为一个标准的Python字典，以便进行序列化
            return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"处理聊天完成时出错: \n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    



if __name__ == "__main__":
    logger.info(f"启动FastAPI应用，监听端口: {PORT}")
    # uvicorn是一个用于运行ASGI应用的轻量级、超快速的ASGI服务器实现
    # 用于部署基于FastAPI框架的异步PythonWeb应用程序
    uvicorn.run(app, host="0.0.0.0", port=PORT)
