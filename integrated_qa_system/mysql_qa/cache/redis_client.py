# Created by erainm on 2025/10/25 17:01.
# IDE：PyCharm 
# @Project: rag_system
# @File：redis_client
# @Description: Redis缓存操作

import redis
from redis import RedisError
import json
from base.config import Config
from base.logger import logger

"""
思路：
    1. 初始化redis连接对象
    2. 实现写入数据通用方法
    3. 实现获取数据通用方法
    4. 实现获取答案的方法
"""

class RedisClient(object):
    def __init__(self):
        try:
            self.config = Config()
            self.logger = logger
            self.redis_client = redis.StrictRedis(host=self.config.REDIS_HOST,
                                                  port=self.config.REDIS_PORT,
                                                  password=self.config.REDIS_PASSWORD,
                                                  db=self.config.REDIS_DB)
            self.logger.info("Redis连接成功")
        except RedisError as e:
            self.logger.error(f"Redis连接失败：{e}")
            raise

    def set_data(self, key, value):
        try:
            self.redis_client.set(key, json.dumps(value))
            self.logger.info(f"Redis写入数据成功：{key} - {value}")
        except RedisError as e:
            self.logger.error(f"Redis写入数据失败：{e}")
            raise

    def get_data(self, key):
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except RedisError as e:
            self.logger.error(f"Redis获取数据失败：{e}")
            return None

    def get_answer(self, query):
        try:
            answer = self.redis_client.get(f'answer:{query}')
            if answer:
                answer = json.loads(answer)
                self.logger.info(f"Redis获取答案成功：{query} - {answer}")
                return answer
        except RedisError as e:
            self.logger.error(f"Redis获取答案失败：{e}")
            return  None

    def set_answer(self, query, answer):
        try:
            self.redis_client.set(f'answer:{query}', json.dumps(answer))
            self.logger.info(f"Redis写入答案成功：{query} - {answer}")
        except RedisError as e:
            self.logger.error(f"Redis写入答案失败：{e}")
            raise
