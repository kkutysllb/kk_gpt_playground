#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-25 18:14
# @Desc   : API接口测试
# --------------------------------------------------------
"""
import requests
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义API端点
url = "http://localhost:8012/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
}

# 默认非流式输出 True or False
stream_flag = False

# 输入文本
input_text = "最贵的套餐是多少钱？"
# input_text = "有没有土豪套餐"
# input_text = "这个套餐是多少钱"
# input_text = "办个200G的套餐"
# input_text = "有没有流量大的套餐"
# input_text = "200元以下，流量大的套餐有啥"
# input_text = "你说那个10G的套餐，叫啥名字"
# input_text = "你说那个100000000G的套餐，叫啥名字"

data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": input_text
        }
    ],
    "stream": stream_flag
}


if __name__ == "__main__":
    if stream_flag:
        # 流式输出
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        # 检查是否为空或不合法的字符串
                        if not json_str:
                            logger.info(f"收到空字符串，跳过...")
                            continue
                        # 确保字符串是有效的JSON格式
                        if json_str.startswith("{") and json_str.endswith("}"):
                            try:
                                json_data = json.loads(json_str)
                                if json_data['choices'][0]['finish_reason'] == 'stop':
                                    logger.info(f"收到终止信号，停止输出...")
                                else:
                                    logger.info(f"流式输出，响应内容是：{json_data['choices'][0]['delta']['content']}")
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解码失败: {e}")
                        
                        else:
                            logger.error(f"无效的JSON格式: {json_str}")
        except Exception as e:
            logger.error(f"流式输出失败: {e}")
            raise f"流式输出失败: {str(e)}"
    else:
        # 非流式输出
        # 发送post请求
        response = requests.post(url, headers=headers, data=json.dumps(data))
        content = response.json()['choices'][0]['message']['content']
        logger.info(f"非流式输出，响应内容是：{content}")
    
