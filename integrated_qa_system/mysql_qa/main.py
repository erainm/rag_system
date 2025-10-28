# Created by erainm on 2025/10/25 17:01.
# IDE：PyCharm 
# @Project: rag_system
# @File：main
# @Description: MySQL系统独立入口，支持查询

from base.logger import logger
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.bm25_search import BM25Search
from mysql_qa.cache.redis_client import RedisClient
import time

class MySQLQASystem:
    def __init__(self):
        self.logger = logger
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(mysql_client=self.mysql_client, redis_client=self.redis_client)

    def query(self, query):
        start_time = time.time()
        self.logger.info(f"开始处理查询：{query}")
        # 执行BM25搜索
        answer, _ = self.bm25_search.search(query, threshold=0.85)
        if answer:
            # 记录BM25查询答案
            self.logger.info(f"BM25查询结果：{answer}")
        else:
            self.logger.info("BM25没有找到答案, 需要调用RAG系统")
            answer = 'BM25未找到答案'
        processing_time = time.time() - start_time
        self.logger.info(f"处理完成，耗时：{processing_time:.2f}秒")
        return answer

def main():
    # 初始化MySQL系统
    mysql_system = MySQLQASystem()
    try:
        print("\n欢迎使用MySQL问答系统")
        print("请输入你的问题，输入‘exit’推出")
        while True:
            query = input("请输入你的问题：").strip()
            if query.lower() == 'exit':
                print("退出系统")
                logger.info("退出MySQL系统")
                break
            answer = mysql_system.query(query)
            print(f"答案：{answer}")
    except Exception as e:
        logger.error(f"处理查询时出错：{e}")
        print(f"处理查询时出错：{e}")
    finally:
        mysql_system.mysql_client.close()

if __name__ == '__main__':
    main()