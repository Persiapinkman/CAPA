from typing import Optional, List, Mapping, Any
from llama_index.core import ServiceContext, SimpleDirectoryReader
from llama_index.core.indices import SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import (
    llm_completion_callback,
)
import requests
import time
import json
import logging
# logger = logging.getLogger('RAG_LOGGER')
import sensenova
sensenova.access_key_id = "5D40F1EFBEFB4593883DA711ACE4C318" #"2Y9AMDj6wCW8XjjxDscJm5YyUrP" 2XqSwZTXGkXPwDkqtG6UsSvnzWT
sensenova.secret_access_key = "3F68AB1FA47E45E995D4E41BE9A6A72D" #"BFAe2OIluVnUCC2ujRHxE03Di7WCtINI" 95MVDylccaffk6ouYKGTa8DteJLLwCgq
API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1RDQwRjFFRkJFRkI0NTkzODgzREE3MTFBQ0U0QzMxOCIsImV4cCI6MTc3MTMxMzUwOCwibmJmIjoxNzM5Nzc3NTAzfQ.kFSejzzjWDBy_y9fUdMELn6TJFoI6uLswtM_52d5f9Q" # 罗氏的一年有效到2026年2月16号

class SenseNovaLLM(CustomLLM):
    model_name: str = 'SenseChat-5'
    # set context window size
    context_window:int = 128048
    # set number of output tokens
    num_output:int = 32048
    return_prompt_as_response: bool = False

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self.return_prompt_as_response:
            return CompletionResponse(text=prompt)
        #response = pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]
        received = False
        try_connect_times = 0
        response = 'time out'
        response = self.sensetime_predict('SenseChat-5', 'None', prompt)

        while not received:
            try:
                response = self.sensetime_predict('SenseChat-5', 'None', prompt)
                received = True
            except Exception as e:
                print(e)
                try_connect_times += 1
                print(f'occur exception, reconnect {try_connect_times} times')
                if try_connect_times > 5:
                    break
            #time.sleep(3)

        # 确保 response 非 None
        if response is None:
            response = "抱歉，模型未能生成有效的回答。(API 错误)"

        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            resp_json = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "SenseChat-5",
                # "model": "DeepSeek-R1",
                "temperature": 0.01,
                "stream": True
            }
            if kwargs:
                resp_json.update(kwargs)
            response = requests.post(
                'https://api.sensenova.cn/v1/llm/chat-completions',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': API_TOKEN
                },
                json=resp_json,
                stream=True
            )

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        json_str = line[5:]
                        try:
                            data = json.loads(json_str)
                            # {'data': {'id': '88df195b-d5cf-4370-ab34-49eb810e0f83', 'usage': {'prompt_tokens': 7, 'completion_tokens': 25, 'knowledge_tokens': 0, 'total_tokens': 32}, 'choices': [{'index': 0, 'role': 'assistant', 'delta': '', 'finish_reason': 'stop'}], 'plugins': {}}, 'status': {'code': 0, 'message': 'OK'}}
                            if 'data' in data and 'choices' in data['data']:
                                content = data['data']['choices'][0]['delta']
                                if len(content) == 0 and data['data']['choices'][0]['finish_reason'] not in ['stop', 'length', '']:
                                    print(f'data:{data}')
                                    content = f'抱歉，模型未能生成有效的回答。({data["data"]["choices"][0]["finish_reason"]})'
                                    yield CompletionResponse(text=content, delta=content)
                                else:
                                    yield CompletionResponse(text=content, delta=content)
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            # logger.error(f"Stream error: {str(e)}")
            yield CompletionResponse(text="流式输出发生错误。")

    def sensetime_predict(self, MODEL, api_secret_key:str, message: str) -> None:
        if MODEL=='SenseChat-5':
            # sensenova.access_key_id="4C688B606A554F6C876D400BBB52B4FF" #"5D40F1EFBEFB4593883DA711ACE4C318"
            # sensenova.secret_access_key ="12489908BEFC46DCAA97A17FFE0BDE18" # "3F68AB1FA47E45E995D4E41BE9A6A72D"
            # print('using SenseChat-5 model')
            # response = sensenova.ChatCompletion.create(
            #     model = "SenseChat-5",
            #     #repetition_penalty=1,
            #     temperature=0.01,
            #     #top_p=0.8,
            #     #top_k=40,
            #     messages = [{"role":"user", "content":message}])
            # print('response',response)
            # sensetime
            # print('你好')

            data = {
                    "messages": [{"role": "user", "content": message}],
                    "model": "SenseChat-5",
                    # "model": "DeepSeek-R1",
                    "temperature": 0.01,
                }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': API_TOKEN
            }

            response = requests.post('https://api.sensenova.cn/v1/llm/chat-completions', headers=headers, json=data)

            if response:
                return response.json()['data']['choices'][0]['message']
            #print(response['data']['choices'][0]['message'])
        elif MODEL=="nova-8k-4x910b":
            print('using nova-ptc-xl-v2-4-0-security-internal-4x910B model')
            #logger.info(message)
            url = 'http://10.139.222.50:31182/generate'
            data = {
                "inputs": f"User: {message} SenseChat:",
                 "parameters":{
                    "temperature": 0.01,
                    "max_new_tokens": 1024,
                    #  "max_new_tokens": 8092,
                     #"top_p": 0.9,
                     #"repetition_penalty": 1.05,

                 }
            }
            headers = {
                'Content-Type': 'application/json',
            }

            response = requests.post(url, headers=headers, json=data)

            if response:
                return response.json()['generated_text']

            # 检查 response 是否为空
            if not response or response.strip() == "":
                response = "抱歉，模型未能生成有效的回答。(API 错误)"  # 或者设置一个默认值

            # 构造 CompletionResponse
            return CompletionResponse(text=response)


if __name__ == "__main__":
    import sys
    model_name='SenseChat-5'
    ctx_window=4000 #int(sys.argv[2])
    nova_llm = SenseNovaLLM(model_name=model_name,context_window=ctx_window,num_output=512)

    # 测试普通对话
    print("测试普通对话:")
    resp = nova_llm.complete('你是谁')
    print(resp)
    exit()

    # 测试流式对话
    print("\n测试流式对话:")
    kwargs = {
        "temperature": 0.01,
        "top_p": 0.7,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.05
    }
    for response in nova_llm.stream_complete('写一篇20字的春天诗歌', **kwargs):
        print(response, end='', flush=True)
        # print(response.delta, end='', flush=True)
    print("\n流式对话测试完成")
