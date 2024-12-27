#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-27 17:55
# @Desc   : 配置文件
# --------------------------------------------------------
"""
import os
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")


MODELS = [
    'qwen2.5-32b-agi',
    'gpt-4o-mini',
    'glm-4-plus'
]

DEFAULT_MODEL = MODELS[0]
MODEL_TO_MAX_TOKENS = {
    'qwen2.5-32b-agi': 4096,
    'gpt-4o-mini': 8192,
    'glm-4-plus': 8192
}


if __name__ == "__main__":
    pass
