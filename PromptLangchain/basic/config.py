#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-26 14:08
# @Desc   : 配置文件
# --------------------------------------------------------
"""

import os
import logging
import time
import uuid
import re
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# 设置日志模板
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


PORT = 8012

# 大模型列表
MODEL_NAME = "qwen-2.5-32b-agi"
LLM_MODELS = ["qwen2.5-32b-agi", "glm-4-plus", "gpt-4o-mini"]

# 温度
TEMPERATURE = 0.3
# 最大输出token数
MAX_TOKENS = 4096
# 频率惩罚
FREQUENCY_PENALTY = 0.0
# 存在惩罚
PRESENCE_PENALTY = 0.0

# API配置
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
ZHIPUAI_API_BASE = os.getenv("ZHIPUAI_API_BASE")

# 定义API端点
url = "http://localhost:8012/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
}

# prompt模板相关
PROMPT_TEMPLATE_SYSTEM = os.path.join(os.path.dirname(__file__), "prompt_template_system.txt")
PROMPT_TEMPLATE_USER = os.path.join(os.path.dirname(__file__), "prompt_template_user.txt")

# 定义messages类
class Messages(BaseModel):
    role:str
    content:str
    

# 定义ChatCompletionRequest类
class ChatCompletionRequest(BaseModel):
    messages: List[Messages]
    stream: Optional[bool] = False

# 定义ChatComplationResponseChoice类
class ChatComplationResponseChoice(BaseModel):
    index: int
    message: Messages
    finish_reason: Optional[str] = None
    

# 定义ChatCompletionResponse类
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list = List[ChatComplationResponseChoice]
    system_fingerprint: Optional[str] = None
    

# 获得模型基础配置
def get_model_config(model_name: str):
    """获取模型配置"""
    api_key = None  # 默认值
    base_url = None  # 默认值
    if model_name == "gpt-4o-mini":
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')
    elif model_name == 'glm-4-plus':
        api_key = os.getenv('ZHIPUAI_API_KEY')
        base_url = os.getenv('ZHIPUAI_API_BASE')
    elif model_name == 'qwen-2.5-32b-agi':
        api_key = os.getenv('LOCAL_API_KEY')
        base_url = os.getenv('LOCAL_API_BASE')
    return api_key, base_url


# 获取prompt在chain中传递的最终结果
def get_prompt(prompt: str):
    logger.info(f"最终给到LLM的prompt: {prompt}")
    return prompt


# 格式化输出，对输出的响应进行段落分割，添加换行符，以及在代码模块中增加，增加输出的可读性
def format_response(response: str):
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    response = str(response)
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用语存储格式化后输出
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和文本交替出现
            parts = re.split(r'```', para)
            for i, part in enumerate(parts):
                # 奇数为代码块
                if i % 2 != 0:
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
            
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace(r'. ', '.\n')
        # 将格式化后的段落添加到formatted_paragraphs列表
        # strip()方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)


if __name__ == "__main__":
    pass
