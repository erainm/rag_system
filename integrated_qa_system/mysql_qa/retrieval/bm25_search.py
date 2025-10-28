# Created by erainm on 2025/10/25 17:02.
# IDE：PyCharm 
# @Project: rag_system
# @File：bm25_search
# @Description: BM25搜索

"""
需求：基于bm25算法实现问题的检索模块，包括：问题加载、分词、BM25 评分、Softmax 归一化，并记录操作日志等。

拆成了两个部分：
1. 数据准备部分：基于mysql、redis中的数据，以及BM25算法类，完成数据加载和算法的初始化
2. 问题检索部分：接收query，计算BM25分值，并返回给用户匹配到的答案

需求：基于mysql、redis中的数据，以及BM25算法类，完成数据加载和算法的初始化
实现：对应__init__和_load_data()方法
思路步骤：
1. 初始化bm25检索器类
    1.1 定义算法类中可能用到的对象： 日志、redis、mysql、bm25算法对象、分词问题列表、原始问题列表
2. 加载问题数据
    2.1 尝试从redis中获取：分词问题列表、原始问题列表
    2.2 如果可以获取到，则直接放到内存里 
    2.3 如果redis获取不到，则通过查询mysql的原始问题并进行以下处理： 原始问题列表 -> 分词 -> 分词问题列表
    2.4 初始化bm25模型，并传入：分词问题列表
"""

import numpy as np
from rank_bm25 import BM25Okapi
from base.logger import logger
from mysql_qa.utils.preprocess import preprocess

class BM25Search(object):
    # 1.1 定义算法类中可能用到的对象： 日志、redis、mysql、bm25算法对象、分词问题列表、原始问题列表
    def __init__(self, mysql_client, redis_client):
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.bm25 = None
        self.tokenized_questions = None
        self.questions = None

        self._loader()

    # 2. 加载问题数据
    def _loader(self):
        # 2.1 尝试从redis中获取：分词问题列表、原始问题列表

        # 原始问题列表
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"

        # 尝试获取原始问题列表
        self.questions = self.redis_client.get_data(original_key)

        # 2.2 如果可以获取到，则直接放到内存里
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 2.3 如果redis获取不到，则通过查询mysql的原始问题并进行以下处理： 原始问题列表 -> 分词 -> 分词问题列表
        if not self.questions:
            # 如果redis里面没有原始问题， 需要从mysql进行加载
            # 形状[多少条数据, 每条数据（含列）]
            all_questions = self.mysql_client.fetch_question()

            if all_questions:
                logger.warn(f'数据库中查不到问题')

            tokenized_questions = [preprocess(question[0]) for question in all_questions]

            self.redis_client.set_data(original_key, [question[0] for question in all_questions])

            self.redis_client.set_data(tokenized_key, tokenized_questions)

        self.tokenized_questions = tokenized_questions

        # TODO bm25必须要接受分词以后得数据 [问题di, [分档的分词]]
        # 2.4 初始化bm25模型，并传入：分词问题列表

        self.bm25 = BM25Okapi(tokenized_questions)

        logger.info('初始化成功')

    def _softmax(self, scores):
        """
        对bm25的得分进行softmax归一化
        :param scores:
        :return:
        """
        # 1. 计算指数
        exp_scores = np.exp(scores - np.max(scores))
        # 2. 计算指数和
        sum_exp_scores = exp_scores.sum()
        # 3. 计算softmax
        softmax_scores = exp_scores / sum_exp_scores
        return softmax_scores

    def search(self, query, threshold=0.85):
        # 搜索查询
        if not query or not isinstance(query, str):
            # 记录无效查询
            self.logger.error("无效查询")
            # 返回 None 和 False
            # None：查询结果是None . False: 是否需要继续调用RAG系统
            return None, False
        # 检查 Redis 缓存
        # 有没有完全一致的问题
        cached_answer = self.redis_client.get_answer(query)
        logger.info(f'缓存中查到了完全一致的问题:{query}')


        if cached_answer:
            # 返回缓存答案
            return cached_answer, False
        try:
            # 分词查询
            query_tokens = preprocess(query)
            # 计算 BM25 分数
            scores = self.bm25.get_scores(query_tokens)
            # 计算 Softmax 分数
            # TODO：把分值(logits)转换成概率
            softmax_scores = self._softmax(scores)
            # 获取最高分索引
            best_idx = softmax_scores.argmax()
            # 获取最高分
            best_score = softmax_scores[best_idx]
            # 检查是否超过阈值
            if best_score >= threshold:
                # 获取原始问题
                original_question = self.questions[best_idx]

                redis_result = self.redis_client.get_answer(original_question)

                # 获取答案
                if redis_result:
                    answer = redis_result
                    logger.info(f'从redis中获取到了问题的答案:{answer}')
                else:
                    answer = self.mysql_client.fetch_answer(original_question)
                    logger.info(f'从mysql中获取到了问题的答案:{answer}')

                if answer:
                    # 缓存答案
                    self.redis_client.set_answer(query, answer)
                    # 记录搜索成功
                    self.logger.info(f"搜索成功，Softmax 相似度: {best_score:.3f}")
                    # 返回答案和 False
                    return answer, False
            # 记录无可靠答案
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            # 返回 None 和 True
            return None, True
        except Exception as e:
            # 记录搜索失败
            self.logger.error(f"搜索失败111: {e}")
            # 返回 None 和 True
            return None, True
