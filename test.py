#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-10-14 21:15
# @Desc   : 设置全局代理
# --------------------------------------------------------
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("LOCAL_API_KEY"),
    base_url=os.getenv("LOCAL_API_BASE"),

)

completion = client.chat.completions.create(
    model="qwen2.5-32b-agi",
    messages=[{"role": "system", "content": "你是一个非常强大，乐于助人的AI小助手，你的名字叫“小智kk”"},
              {"role": "user", "content": "你好，小智kk，能介绍你自己吗？同时，我想知道北京今天的天气怎么样？"},],
)

if __name__ == "__main__":
    print(completion.choices[0].message.content)