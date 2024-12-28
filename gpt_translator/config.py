#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-27 21:47
# @Desc   : 配置文件
# --------------------------------------------------------
"""
import os
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

API_KEY = os.getenv("ZHIPUAI_API_KEY")
BASE_URL = os.getenv("ZHIPUAI_API_BASE")

EMBEDDINGS_MODEL = "embedding-3"


MODELS = [
    'glm-4-plus',
    'glm-4v',
    'glm-4v-plus',
]

DEFAULT_MODEL = MODELS[0]
MODEL_TO_MAX_TOKENS = {
    'glm-4-plus': 8192,
    'glm-4v': 8192,
    'glm-4v-plus': 8192
}

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEFAULT_MAX_TOKENS = 4096

if __name__ == "__main__":
    pass
