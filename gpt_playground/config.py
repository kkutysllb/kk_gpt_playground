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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")





if __name__ == "__main__":
    pass
