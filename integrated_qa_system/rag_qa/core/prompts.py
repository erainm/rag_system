# Created by erainm on 2025/10/26 20:19.
# IDE：PyCharm 
# @Project: rag_system
# @File：prompts
# @Description: 

# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain.prompts import PromptTemplate


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    # 定义 RAG 提示模板
    @staticmethod
    def rag_prompt():
        '''添加了历史记录，注意用在：new_rag_system'''
        return PromptTemplate(
            template="""
            你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：
    
            1. 分析问题和上下文：
               - 基于提供的上下文（如果有）和你的知识回答问题。
               - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档，……”。
    
            2. 评估对话历史：
               - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
               - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
               - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。
    
            3. 生成回答：
               - 提供清晰、准确的回答，避免无关信息。
               - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
            对话历史:
             {history}
            上下文:
             {context}
            问题:
             {question}
    
            回答:
            """,
            # 定义输入变量
            input_variables=["context", "history", "question", "phone"],
        )

    # 定义假设问题生成的Prompt模版
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
               问题: {query}  
               假设答案:  
               """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               将以下复杂查询分解为多个简单子查询，每行一个子查询，最多生成两个子查询（只保留子查询问题，其他的文本都不需要）：
               eg: 
               用户原始query："Milvus 和 Zilliz Cloud 在功能上有什么不同？
               子查询："Milvus 有哪些功能？"，"Zilliz Cloud 有哪些功能？"

               查询: {query}  
               子查询:  
               """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               将以下复杂查询简化为一个更简单的问题：  
               查询: {query}  
               简化问题:  
               """,
            #   定义输入变量
            input_variables=["query"],
        )


if __name__ == '__main__':
    # rga_prompt = RAGPrompts.rag_prompt()
    # result = rga_prompt.format(context="黑马程序员", question="这个机构叫什么名称", phone="12345", history="")
    # print(f'result-->{result}')
    hyde = RAGPrompts.subquery_prompt()
    result = hyde.format(query="AI和JAVA各自有什么作用")
    print(result)