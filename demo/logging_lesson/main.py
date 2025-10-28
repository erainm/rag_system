# Created by erainm on 2025/10/25 15:23.
# IDE：PyCharm 
# @Project: rag_system
# @File：main
# @Description: 日志主程序


from utils.logger import set_logger

logger = set_logger("MainApp")

def process_data(data):
    logger.info("Processing data...")
    if not data:
        logger.error("数据为空，无法处理数据")
        return None
    logger.info("数据处理完成")
    return data.upper()

def main():
    logger.info("程序启动")
    result = process_data("hello world")
    if result:
        logger.info("处理结果：{}".format(result))
    else:
        logger.warning("处理失败")
    logger.info("程序结束")

if __name__ == '__main__':
    main()