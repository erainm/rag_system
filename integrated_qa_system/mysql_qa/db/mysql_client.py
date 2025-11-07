# Created by erainm on 2025/10/25 17:02.
# IDE：PyCharm 
# @Project: rag_system
# @File：mysql_client
# @Description: MySQL数据库操作

import pymysql
import pandas as pd
from base.config import conf
from base.logger import logger

class MySQLClient(object):
    def __init__(self):
        self.logger = logger
        self.conn = conf
        try:
            self.conn = pymysql.connect(
                host=conf.MYSQL_HOST,
                port=conf.MYSQL_PORT,
                user=conf.MYSQL_USER,
                password=conf.MYSQL_PASSWORD,
                database=conf.MYSQL_DATABASE,
                charset='utf8mb4'
            )
            # 创建游标
            self.cursor = self.conn.cursor()
            # 记录连接成功
            self.logger.info("MySQL数据库连接成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL数据库连接失败：{e}")
            raise

    # 创建表
    def create_table(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS `jpkb` (
          `id` INT AUTO_INCREMENT PRIMARY KEY,
          `subject_name` varchar(20),
          `question` varchar(1000),
          `answer` text) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        try:
            self.cursor.execute(create_table_query)
            self.conn.commit()
            self.logger.info(f"MySQL表创建成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL表创建失败：{e}")
            raise

    # 插入数据，读取csv插入mysql
    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            count = 0
            for _, row in data.iterrows():
                # row -> 类似于dict的格式
                # pred-statement
                count += 1
                insert_query = "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))
            self.conn.commit()
            logger.info(f"数据插入成功:{count}行")
        except Exception as e:
            self.conn.rollback()
            logger.error(f'输入插入失败:{e}')

    # 获取所有问题
    def fetch_question(self):
        # 获取所有问题
        try:
            select_query = "SELECT question FROM jpkb"
            self.cursor.execute(select_query)
            # 获取结果
            questions = self.cursor.fetchall()
            logger.info(f"获取所有MySQL问题成功:{len(questions)}行")
            return questions
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL获取问题失败：{e}")
            return []

    # 获取答案
    def fetch_answer(self, question):
        # 获取指定问题答案
        try:
            sql = "SELECT answer FROM jpkb WHERE question = %s"
            self.cursor.execute(sql, (question,))
            answer = self.cursor.fetchone()
            return answer[0] if answer else None
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL获取答案失败：{e}")
            return None

    # 关闭连接
    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
            self.logger.info("MySQL数据库连接已关闭")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL数据库关闭失败：{e}")

if __name__ == '__main__':
    mysql_client = MySQLClient()
    # mysql_client.create_table()
    # mysql_client.insert_data('/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/mysql_qa/data/JP学科知识问答.csv')
    # answer = mysql_client.fetch_answer('爬虫爬取下来的数据写入文件有哪些方式')
    all_questions = mysql_client.fetch_question()
    print(all_questions)