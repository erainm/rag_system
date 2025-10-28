# Created by erainm on 2025/10/25 17:00.
# IDE：PyCharm 
# @Project: rag_system
# @File：logger
# @Description: 日志设置


"""
    1. 确保日志目录存在，不存在就创建
    2. 创建日志记录器：Logger
        2.1 获取Logger对象
        2.2 设置日志级别为所控制器最低的（设置全局日志级别）
    3. 创建控制台控制器：StreamHandler
        3.1 创建控制台处理器对象
        3.2 设置日志级别为INFO
    4. 创建文件处理器：FileHandler，并指定目录
        4.1 创建文件处理对象
        4.2 设置日志级别为DEBUG
    5. 定义并设置日志格式：
        5.1 定义日志格式：logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        5.2 设置处理器日志格式
    6. 把处理器添加到logger中
"""
import os.path
import logging
import sys
from base.config import conf


def setup_logger(logger_name='EDU_RAG', log_file=conf.LOG_FILE):
    dir_name = os.path.dirname(log_file)
    # 1. 确保日志目录存在，不存在就创建
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # 2. 创建日志记录器：Logger
    # 2.1 获取Logger对象
    logger = logging.getLogger(logger_name)
    # 2.2 设置日志级别为所控制器最低的（设置全局日志级别）
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        # 3. 创建控制台控制器：StreamHandler
        # 3.1 创建控制台处理器对象
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # 3.2 设置日志级别为INFO
        stream_handler.setLevel(logging.INFO)
        # 4. 创建文件处理器：FileHandler，并指定目录
        #     4.1 创建文件处理对象
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        #     4.2 设置日志级别为DEBUG
        file_handler.setLevel(logging.INFO)
        # 5. 定义并设置日志格式：
        #     5.1 定义日志格式：logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        #     5.2 设置处理器日志格式
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # 6. 把处理器添加到logger中
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return  logger


logger = setup_logger('EDU_RAG')
