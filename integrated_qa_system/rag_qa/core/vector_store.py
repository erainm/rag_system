# Created by erainm on 2025/10/26 16:33.
# IDE：PyCharm 
# @Project: rag_system
# @File：vector_store
# @Description: Milvus交互

"""
    1. 初始化与集合管理： 创建及加载Milvus向量数据库集合
        1.1 初始化方法：初始化VectorStore类的实例，设置基本参数并调用集合创建或加载方法
        1.2 创建或加载集合方法：检查并创建或加载Milvus集合，定义字段结构和索引参数
    2. 文档向量化与存储： 将分块后的文档转换为向量并存储
        添加文档方法：将分块后的文档转换为向量并存储到Milvus集合
    3. 混合检索与重排序： 结合稠密和稀疏向量进行检索，并通过重排序优化结果
"""
import os.path
import sys

# 导入模型加载器： BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain.docstore.document import Document
# hugging-face开源的基于transformer架构的开源模型库，专门用于处理段落；导入 CrossEncoder，交叉学习，用于重排序和 NLI 判断，加载rerank模型
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
from rag_qa.core.document_processor import *
from base.logger import logger
from base.config import conf

# 获取当前文件所在目录的绝对路径(找到core这一层)
current_dir = os.path.abspath(os.path.dirname(__file__))
print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径（找到rag_qa这一层）
rag_qa_path = os.path.dirname(current_dir)
print(f'rag_qa_path--》{rag_qa_path}')
# 添加系统路径
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

"""
    实现初始化方法：初始化VectorStore类的实例，设置基本参数并调用集合创建或加载方法
    1. 初始化Milvus向量数据库
        1.1 用conf中的默认初始化集合名称、主机、端口和数据库名称
        1.2 把collection索引加载到内存中
    2. 初始化模型
        2.1 rerank：加载BEG-Rerank模型，用于后续重排序
        2.2 embedding_function：初始化BGE-M3嵌入模型，禁用FP16，使用CPU运行
        2.3 dense_dim：获取稠密向量的维度
"""
# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        # 检查CUDA是否可用
        self.device = 'cpu'
        # 日志提醒使用的是什么设备
        self.logger.info(f"使用设置：{self.device}")
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        print(f'reranker_path--》{reranker_path}')
        # rerank模型加载，从milvus查询到context（多个父块）后，再根据context和query的关联做一个重排序
        self.reranker = CrossEncoder(reranker_path, device=self.device)
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        beg_m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(model_name_or_path=beg_m3_path, use_fp16=False, device=self.device)
        # 获取稠密向量的维度 1024
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端（先不指定数据库）
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 检查并创建数据库
        self._create_database_if_not_exists()
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    def _create_database_if_not_exists(self):
        """检查并创建数据库"""
        try:
            # 获取现有数据库列表
            db_list = self.client.list_databases()
            if self.database not in db_list:
                # 创建数据库
                self.client.create_database(db_name=self.database)
                self.logger.info(f"已创建数据库 {self.database}")
            else:
                self.logger.info(f"已存在数据库 {self.database}")
        except Exception as e:
            self.logger.error(f"创建数据库失败: {e}")
            raise

    """
    创建或加载集合方法：检查并创建或加载Milvus集合，定义字段结构和索引参数
    1. 判断集合是否存在，若存在则进行加载
    2. 集合不存在，创建新集合
        2.1 定义集合各字段 id、text、dense_vector、sparse_vector、parent_id、parent_content、source、timestamp
        2.2 定义索引：稠密向量索引：dense_vector,稀疏向量索引：sparse_vector
    3. 把集合的索引加载到内存中
    """
    def _create_or_load_collection(self):
        # 检查指定集合是否已经存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                # drop_ratio_build,在构建索引时，按一定比例丢弃向量中绝对值较小的元素
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"集合已存在 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询，相当于构建索引，让milvus的这个表可以进行向量匹配查询
        self.client.load_collection(self.collection_name)

    """
    文档向量化与存储： 将分块后的文档转换为向量并存储
        1. 提取文本，从文档对象中提取文本内容
        2. 生成向量，使用BGE-M3模型生成稠密向量和稀疏向量
        3. 构造数据，为每篇文档生成唯一ID（MD5哈希），将向量和元数据组成字典
        4. 使用upsert操作插入或更新数据
    """
    def add_documents(self, documents: list[Document]):
        """
        :param documents: 已经处理成子块的数据，Document是一个子块
        """
        # print(f'documents--》{documents[0]}')
        # 提取所有文档的内容列表
        texts = [doc.page_content for doc in documents]

        # 使用 BGE-M3 嵌入函数生成文档的嵌入
        # TODO： embeddings['dense'] -> dense_vector:一个文本用一个1024的向量表示，dense_vector [文档id, 文本向量]
        # TODO： embeddings['sparse'] -> sparse_vector:一个文本用一个稀疏向量表示，sparse_vector {单词：单词对应权重， XXXXXXXX}
        embeddings = self.embedding_function(texts)
        # print(f'embeddings--》{embeddings}')
        # print(f'embeddings--》{embeddings.keys()}')
        # 初始化空列表，存储插入的数据
        data = []
        # 遍历每个文档，带上索引index
        for index, doc in enumerate(documents):
            # 生成文档内容的哈希值作为唯一的ID
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # print(f'text_hash--》{text_hash}')
            # print(f'text_hash--》{type(text_hash)}')
            # 初始化一个稀疏向量的字典（Milvus要求存储稀疏向量的格式）
            sparse_vector = {}
            # 获取第index行对应的稀疏向量数据[0.4, 0.2, 0, 0, 0.1]
            print(f'row--》{embeddings["sparse"]}')
            # row = embeddings["sparse"][index, :]
            row = embeddings['sparse'][[index], :]
            # row = embeddings["sparse"][i]:新版本milvus-model，支持这种获取稀疏向量的形式
            # indics = row.row
            # print(f'row--》{row}')
            # print(f'row--》{row.shape}')
            # 获取稀疏向量的非零值的索引,模型返回的结果 col和data是分开存储的，需要转换为 {单词id：单词权重}
            indics = row.indices
            # print(f'indics--》{indics}')
            # 获取稀疏向量的非零值
            values = row.data
            # 将索引和值进行配对，存储到字典中
            for token_id, value in zip(indics, values):
                sparse_vector[token_id] = value
            # print(f'sparse_vector--》{sparse_vector}')
            # print(f'sparse_vector--》{len(sparse_vector)}')
            # print(embeddings["dense"][i])
            # print(embeddings["dense"][i].shape)
            # 创建数据字典，包含所有字段
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                # 稠密向量: [文档id(第几个子块)， 文本向量] (传入文档id) -> 当前这个文档对应的文本向量
                "dense_vector": embeddings["dense"][index],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })
        # 检查是否有数据需要插入
        if data:
            # 使用 upsert 操作插入数据，覆盖重复 ID
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录插入或更新的文档数量日志
            logger.info(f"已插入或更新Milvus: {len(data)} 个文档")
        else:
            # 如果没有数据需要插入，则记录日志
            logger.info("没有文档数据需要插入Milvus")


    """
    混合检索与重排序： 结合稠密和稀疏向量进行混合检索，并通过重排序优化结果
        1. 生成查询向量：使用BGE- M3生成稠密和稀疏向量
        2. 构造检索请求（混合检索）
            2.1 构造稠密向量的AnnSearchRequest
            2.2 构造稀疏向量的AnnSearchRequest
        3. 混合检索：使用WeightedRanker融合结果
        4. 重排序，使用CrossEncoder：rerank重新排序父文档
    """
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([query])
        # 获取查询的稠密向量
        # print(f'query_embeddings---》{query_embeddings}')
        dense_query_vector = query_embeddings["dense"][0]
        # print(f'dense_query_vector--》{dense_query_vector.shape}')
        # 初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # 获取查询稀疏向量的第 0 行数据
        # row = query_embeddings["sparse"][0, :]
        row = query_embeddings['sparse'][[0],:]
        # 获取稀疏向量的非零值索引
        indices = row.indices
        # 获取稀疏向量的非零值
        values = row.data
        # 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value
        # print(f'sparse_query_vector-->{sparse_query_vector}')
        # 初始化过滤表达式，默认不过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        # print(f'filter_expr--》{filter_expr}')
        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        # 创建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.7，稠密向量权重 1.0
        ranker = WeightedRanker(1.0, 0.7)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]
        # print(f'results--》{results}')
        # print(f'results--》{type(results)}')
        # print(f'results--》{len(results)}')
        # 将上述搜索到的结果进行Document对象封装，便于查询使用
        sub_chunks = [self._doc_from_hit(hit["entity"])for hit in results]
        # print(f'sub_chunks--》{len(sub_chunks)}')
        # 从子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # print(f'parent_docs--》{parent_docs}')
        # print(f'parent_docs--》{len(parent_docs)}')
        # # 如果只有1个文档或者没有，直接返回跳过重排序
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]
            # 如果有父文档，进行重排序
        if parent_docs:
            # 如果父块只有一个，进行返回,这里的parent_docs就是context
            if len(parent_docs) < 2:
                return parent_docs
            # 如果父块超过一个，需要进行重排序，基于query和context的匹配程度做重排序
            # 创建查询与文档内容的配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用 BGE-Reranker 计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # print(f'scores--》{scores}')
            # 根据得分从高到低排序文档，使用索引排序而不是直接排序Document对象
            scored_docs = list(zip(scores, parent_docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)  # 按得分排序
            ranked_parent_docs = [doc for score, doc in scored_docs]
        # 如果没有父文档，返回空列表
        else:
            ranked_parent_docs = []

        # 返回前 m 个重排序后的文档
        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
            # 返回去重后的父文档列表
        return unique_docs

    # 定义类似私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )

if __name__ == '__main__':
    vector_store = VectorStore()
    # directory_path = '/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/data/ai_data'
    # print(f"embedding_function.dim--》{vector_store.embedding_function.dim}")
    # documents = process_documents(directory_path)
    # vector_store.add_documents(documents)
    query = "windows电脑怎么安装redis"
    results = vector_store.hybrid_search_with_rerank(query, source_filter='ai')
    print(f'results-->{results}')
    print(f'results-->{len(results)}')