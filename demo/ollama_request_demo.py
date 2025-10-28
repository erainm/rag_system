# Created by erainm on 2025/10/24 20:39.
# IDE：PyCharm 
# @Project: rag_system
# @File：ollama_request_demo
# @Description: 
# TODO:

import requests
import json

host='127.0.0.1'
port='11434'
url=f'http://{host}:{port}/api/chat'
model='qwen3:4b'
headers={
    'Content-Type': 'application/json'
}
data={
    'model': model,
    'options':{'temperature':0.0},
    'stream': False,
    'messages': [
        {'role': 'user', 'content': '天空为什么是蓝色的？'}
    ]
}
response = requests.post(url, headers=headers, json=data)
res=response.json()
print(json.dumps(res))