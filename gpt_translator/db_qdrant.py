#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-27 23:01
# @Desc   : 数据库操作
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from qdrant_client.http.exceptions import UnexpectedResponse  # 捕获错误信息
from config import QDRANT_HOST, QDRANT_PORT


class QdrantDB:
    def __init__(self) -> None:
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)  # 创建客户端实例
        self.size = 2048  # openai embedding 的维度是2048
        
    def get_points_count(self, collection_name):
        """
        先检查集合是否存在。
        - 如果集合存在，返回该集合的 points_count （集合中确切的points_count）
        - 如果集合不存在，创建集合。
            - 创建集合成功，则返回 points_count （0: 刚创建完points_count就是0）
            - 创建集合失败，则返回 points_count （-1: 创建失败了，定义points_count为-1）

        Returns:
            points_count

        Raises:
            UnexpectedResponse: 如果在获取集合信息时发生意外的响应。
            ValueError: Collection test_collection not found
        """
        try:
            collection_info = self.get_collection(collection_name)
        except (UnexpectedResponse, ValueError) as e:  # 集合不存在，则创建新的集合
            if self.create_collection(collection_name):
                logger.success(f"创建集合成功 | collection_name：{collection_name} points_count: 0")
                return 0
            else:
                logger.error(f"创建集合失败 | collection_name：{collection_name} 错误信息: {e}")
                return -1
        except Exception as e:
            logger.error(f"获取集合信息时发生错误 | collection_name：{collection_name} 错误信息:{e}")
            return -1  # 返回错误码或其他适当的值
        else:
            points_count = collection_info.points_count
            logger.success(f"库里已有该集合 | collection_name：{collection_name} points_count：{points_count}")
            return points_count
        
        
    def list_all_collections_names(self):
        """
        CollectionsResponse类型举例：
        CollectionsResponse(collections=[
            CollectionDescription(name='GreedyAIEmployeeHandbook'),
            CollectionDescription(name='python')
        ])
        CollectionsResponse(collections=[])
        """ 
        CollectionsResponse = self.client.get_collections()
        collections_names = [collection.name for collection in CollectionsResponse.collections]
        return collections_names
    
    
    def create_collection(self, collection_name):
        """
        创建集合。

        Args:
            collection_name (str, optional): 自定义的集合名称。如果未提供，则使用默认的self.collection_name。

        Returns:
            bool: 如果成功创建集合，则返回True；否则返回False。
        """
        return self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.size, distance=Distance.COSINE)
        )
    
    
    def get_collection(self, collection_name):
        """
        获取集合信息。

        Args:
            collection_name (str, optional): 自定义的集合名称。如果未提供，则使用默认的self.collection_name。

        Returns:
            collection_info: 集合信息。
        """
        collection_info = self.client.get_collection(collection_name)
        return collection_info
    
    
    def add_points(self, collection_name, vectors, payloads):
        # 将数据点添加到Qdrant
        self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=Batch(
                ids=list(range(1, len(vectors) + 1)),
                vectors=vectors,
                payloads=payloads
            )
        )
        return True
    
    
    def search(self, collection_name, query_vector, limit=3):
        """
        搜索与查询向量最相似的点
        """
        # 搜索与查询向量最相似的点
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return search_results
    
    
    def get_collection_content(self, collection_name, limit=1000):
        """
        获取ScoredPoint对象列表
        """
        score_points = self.client.search(
            collection_name=collection_name,
            query_vector=[0.0] * self.size,
            limit=limit,
            with_payload=True
        )
        
        # 将该对象列表按id升序排序
        score_points.sort(key=lambda x: x.id)
        logger.info(f"当前集合：{collection_name} 的节点总数：{len(score_points)}")
        
        # 提取每个ScoredPoint对象中的payload字典中的page_content
        # payload表示向量的附加信息，每个payload都是一个字典，包含了page_content和metadata）
        page_contents = [point.payload.get('page_content', '') for point in score_points]
        content = "".join(page_contents)
        logger.trace(f"当前集合：{collection_name} 的内容字符数：{len(content)}")
        return content






if __name__ == "__main__":
    # 测试
    qdrant = QdrantDB()
    
    collection_names = qdrant.list_all_collections_names()
    
    # 获取集合信息
    # qdrant.get_collection(collection_name)
    # 如果之前没有创建集合，则会报以下错误
    # qdrant_client.http.exceptions.UnexpectedResponse: Unexpected Response: 404 (Not Found)
    # Raw response content:
    # b'{"status":{"error":"Not found: Collection `test_collection` doesn\'t exist!"},"time":0.000198585}'

    # 获取集合信息，如果没有该集合则创建
    # count = qdrant.get_points_count(collection_name)
    # print(count)
    # 如果之前没有创建集合，且正确创建了该集合，则输出0。例：创建集合成功。集合名：test_collection。节点数量：0。
    # 如果之前创建了该集合，则输出该集合内部的节点数量。例：库里已有该集合。集合名：test_collection。节点数量：100  0。
    
    # 查询集合内容
    # content = qdrant.get_collection_content(collection_name)
    # print(content)
    
    # 删除集合
    for collection_name in collection_names:
        qdrant.client.delete_collection(collection_name)
