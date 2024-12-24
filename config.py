#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-10-14 21:15
# @Desc   : 配置文件
# --------------------------------------------------------
"""

import os

# 大模型列表
LLM_MODELS = ["qwen2.5-32b-agi", "glm-4-plus", "gpt-4o-mini"]

# API配置
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
ZHIPUAI_API_BASE = os.getenv("ZHIPUAI_API_BASE")
