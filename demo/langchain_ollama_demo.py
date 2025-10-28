# Created by erainm on 2025/10/24 20:20.
# IDE：PyCharm 
# @Project: rag_system
# @File：langchain_ollama_demo
# @Description: langchain 调用ollama
# TODO:

from langchain_community.llms import ollama

host='127.0.0.1'
port='11434'
llm = ollama.Ollama(base_url=f'http://{host}:{port}', model='qwen3:4b',temperature=0)

res=llm.invoke('你是谁？')
print(res)
