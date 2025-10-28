# Created by erainm on 2025/10/27 17:13.
# IDE：PyCharm 
# @Project: rag_system
# @File：rag_evaluate
# @Description: RAG系统评估

# 导入pandas库，用于数据处理和保存CSV文件
import pandas as pd

from base.config import conf
# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
    ContextRecall
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入langchain_community的Ollama聊天模型和嵌入模型，用于本地模型调用
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
# 导入json库，用于加载JSON格式的评估数据集
import json

# 1. 加载生成的数据集
# 使用with语句打开JSON文件，确保文件正确关闭，指定编码为utf-8
with open("/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/rag_assesment/rag_evaluate_data.json", "r", encoding="utf-8") as f:
    # 将JSON文件内容加载到data变量中，data为包含多个数据条目的列表
    data = json.load(f)

# print(f'data--》{data}')
print(f'data--》{len(data)}')
# 2. 转换为RAGAS格式
# 创建字典eval_data，将JSON数据转换为RAGAS要求的字段格式
eval_data = {
    # 提取每个数据条目的question字段，组成问题列表
    "question": [item["question"] for item in data],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [item["context"] for item in data],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data]
}
# print(eval_data)
# 使用Dataset.from_dict将字典转换为RAGAS所需的Dataset对象
dataset = Dataset.from_dict(eval_data)
print(f'dataset--》{dataset}')

# 3. 配置RAGAS评估环境
# 初始化千问模型
llm = ChatOpenAI(
    model_name=conf.LLM_MODEL,
    openai_api_base=conf.DASHSCOPE_BASE_URL,
    openai_api_key=conf.DASHSCOPE_API_KEY,
    temperature=0.1
)
# 初始化嵌入千问模型，用于计算语义相似度
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=conf.DASHSCOPE_API_KEY
)

# embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# # 初始化ChatOllama模型，需要选择一个模型，设置ollama服务地址
# llm = ChatOllama(model="qwen3:4b", base_url='http://localhost:11434')
# embeddings = OllamaEmbeddings(model="qwen2.5:7b",base_url='http://localhost:11434' )
# embeddings = OllamaEmbeddings(model="bge-m3",base_url='http://localhost:11434')


# 4. 执行评估
# 调用evaluate函数，传入数据集、评估指标、LLM模型和嵌入模型
result = evaluate(
    # 传入转换好的Dataset对象
    dataset=dataset,
    # 指定使用的评估指标列表
    metrics=[
        Faithfulness(),  # 忠实度：答案是否基于上下文
        AnswerRelevancy(),  # 答案相关性：答案与问题的匹配度
        ContextRelevance(),  # 上下文相关性：上下文是否仅包含相关信息
        ContextRecall()  # 上下文召回率：上下文是否包含所有必要信息
    ],
    # 传入配置好的LLM模型
    llm=llm,
    # 传入配置好的嵌入模型
    embeddings=embeddings
)

# 5. 输出并保存结果
# 打印评估结果标题
print("RAGAS评估结果：")
# 打印评估结果，包含各指标的分数
print(result)
# 将评估结果转换为pandas DataFrame 便于保存
# 将结果转换为字典格式再创建DataFrame
try:
    if isinstance(result, list):
        # 如果 result 是列表，检查列表内容
        if len(result) > 0:
            print(f"First item type: {type(result[0])}")
            # 如果列表包含字典或其他可处理的对象
            df = pd.DataFrame(result)
        else:
            # 空列表情况
            df = pd.DataFrame([{"result": "empty list"}])
    elif hasattr(result, 'scores') and isinstance(result.scores, dict):
        # 如果是 Score 对象且 scores 是字典
        result_dict = {
            'faithfulness': result.scores.get('faithfulness', 0) if isinstance(result.scores, dict) else 0,
            'answer_relevancy': result.scores.get('answer_relevancy', 0) if isinstance(result.scores, dict) else 0,
            'context_relevancy': result.scores.get('context_relevancy', 0) if isinstance(result.scores, dict) else 0,
            'context_recall': result.scores.get('context_recall', 0) if isinstance(result.scores, dict) else 0
        }
        df = pd.DataFrame([result_dict])
    elif isinstance(result, dict):
        # 如果是字典格式
        df = pd.DataFrame([result])
    else:
        # 其他情况，创建包含结果字符串的DataFrame
        df = pd.DataFrame([{"result": str(result), "type": str(type(result))}])
except Exception as e:
    print(f"处理结果时出错: {e}")
    # 最后的备选方案
    df = pd.DataFrame([{"result": str(result)}])

# 将DataFrame保存为CSV文件，不保存索引
df.to_csv("rag_evaluate_result.csv", index=False)