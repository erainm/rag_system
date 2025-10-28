# Created by erainm on 2025/10/25 10:54.
# IDE：PyCharm 
# @Project: rag_system
# @File：milvus_demo
# @Description: 
# TODO:
from pymilvus import MilvusClient, DataType


# milvus 数据库操作
def operate_milvus_db():
    client = MilvusClient(url='http://127.0.0.1:19530')
    # 创建milvus_demo数据库
    databases = client.list_databases()
    if 'milvus_demo' not in databases:
        client.create_database(db_name='milvus_demo')
    else:
        client.using_database(db_name='milvus_demo')
    return  client

# 创建集合collection
def create_collection():
    # 定义schema
    """
        注意：在定义集合schema时，enable_dynamic_field=True 使得可以插入未定义的字段，一般动态字段以JSON格式存储，通常命名为$meta，在插入数据时，所有未定义的字段及其值
        在定义集合schema时，auto_id=True 可以对主键自增增长id
    :return:
    """
    client = operate_milvus_db()
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    # schema添加字段：id， vector
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name='scalar1', datatype=DataType.VARCHAR, max_length=256, description='标量字段')

    client.create_collection(collection_name='demo_v1', schema=schema)

    # 设置索引
    index_params = client.prepare_index_params()
    index_params.add_index(field_name='vector', index_type='', metric_type='COSINE', index_name='vector_index')
    client.create_index(collection_name='demo_v1', index_params=index_params)

    # 查看索引信息
    res = client.list_indexes(collection_name='demo_v1')
    print(f'索引信息 ---> {res}')

    res = client.describe_collection(collection_name='demo_v1', index_name='vector_index')
    print(f'指定索引详细信息 ---> {res}')




if __name__ == '__main__':
    # client = operate_milvus_db()
    create_collection()