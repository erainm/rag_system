# Created by erainm on 2025/10/24 20:01.
# IDE：PyCharm 
# @Project: rag_system
# @File：ollama_demo
# @Description: ollama练习
# TODO:

import ollama

def ollama_demo():
    response = ollama.chat(
        model= 'qwen3:4b',
        messages=[
            {'role': 'user', 'content': '天空为什么是蓝色的？'}
        ]
    )

    print(response)
    print(response['message']['content'])


if __name__ == '__main__':
    ollama_demo()