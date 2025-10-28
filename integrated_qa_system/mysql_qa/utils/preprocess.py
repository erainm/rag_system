# Created by erainm on 2025/10/25 17:02.
# IDE：PyCharm 
# @Project: rag_system
# @File：preprocess
# @Description: 文本预处理

import jieba
from base.logger import logger

def preprocess(text):
    """
    对文本进行预处理，包括分词、去停用词、去标点符号等
    :param text: 输入的文本
    :return: 处理后的文本
    """
    logger.info("开始进行文本预处理...")
    try:
        logger.info("文本处理完成")
        return jieba.lcut(text.lower())
    except Exception as e:
        logger.error("文本预处理出错：{}".format(e))
        return []