# Created by erainm on 2025/10/25 16:59.
# IDE：PyCharm 
# @Project: rag_system
# @File：config
# @Description: 配置管理，加载config.ini
import configparser

class Config:
    # 初始化配置， 加载config.ini文件
    def __init__(self, config_file: str = "/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/config.ini"):
        # 创建配置解析器
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(config_file)

        # MySQL配置
        self.MYSQL_HOST = self.config.get("mysql", "host", fallback="localhost")
        self.MYSQL_PORT = self.config.getint("mysql", "port", fallback=3306)
        self.MYSQL_USER = self.config.get("mysql", "user", fallback="root")
        self.MYSQL_PASSWORD = self.config.get("mysql", "password", fallback="123456")
        self.MYSQL_DATABASE = self.config.get("mysql", "database", fallback="subjects_kg")

        # Redis配置
        self.REDIS_HOST = self.config.get("redis", "host", fallback="localhost")
        self.REDIS_PORT = self.config.getint("redis", "port", fallback=6379)
        self.REDIS_PASSWORD = self.config.get("redis", "password", fallback='1234')
        self.REDIS_DB = self.config.getint("redis", "db", fallback=0)

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        # Milvus 端口
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='subjects_kg')
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edu_rag')

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model_name')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone')
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback=["ai", "java", "test", "ops", "bigdata"]))

        # 日志文件路径
        self.LOG_FILE = self.config.get("logger", "log_file", fallback="/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/logs/app.log")

conf = Config()

if __name__ == '__main__':
    config = Config()
    print(config.VALID_SOURCES)