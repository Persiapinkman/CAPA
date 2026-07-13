import json
import os

import requests
import sensenova
from loguru import logger
from typing import Callable
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from openai import AzureOpenAI
import openai_proxy

def api2d_generate(**params) -> str:
    prompter = params.get('prompter', '{message}')
    max_retries = params.get('max_retries', 1)

    message = prompter.format(**params)
    url = "https://oa.api2d.net/v1/chat/completions"
    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "safe_mode": False
    })
    headers = {
        'Authorization': 'Bearer fk203645-mmIWQ9MHGyUOWRzMeczPVOD3NTHwIKSl',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    success = False
    num_retries = 0
    if max_retries > 1:
        while not success or num_retries < max_retries:
            try:
                response = requests.request("POST", url, headers=headers, data=payload).json()
                response = response['choices'][0]['message']['content']
                success = True
            except Exception as e:
                success = False
                logger.error(f'error: {e} | num_retries: {num_retries+1}')
            num_retries += 1
        return response
    else:
        response = requests.request("POST", url, headers=headers, data=payload).json()
        response = response['choices'][0]['message']['content']
        return response



def nova_generate(message: str) -> str:
    sensenova.access_key_id = os.environ.get("SENSENOVA_ACCESS_KEY_ID", "2SNCoPQ1VwQdn1Re8BO1hXxd6v4")
    sensenova.secret_access_key = os.environ.get("SENSENOVA_SECRET_ACCESS_KEY", "pUTYti9G94Xvtwckx22HwqFKMn76ECoC")
    params = {
        # "model_id": 'nova-ptc-xl-v1.2.1-0810',
        "model": 'nova-ptc-xl-v2-internal-test',
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.8,
        "top_p": 0.7,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.05,
        "stream": False,
        "user": "test"
    }
    resp = sensenova.ChatCompletion.create(
            model=params.get('model_id', 'nova-ptc-xl-v2-internal-test'), #nova-ptc-xl-v1
            max_new_tokens=params.get('max_new_tokens', 1024),
            repetition_penalty=params.get('repetition_penalty', 1.05),
            stream=params.get('streaming', False),
            temperature=params.get('temperature', 0.1),
            top_p=params.get('top_p', 0.7),
            messages=message
        )
    return resp

def nova_v2_generate(message):
    url = 'https://sensenova.sensetime.com/v1/nlp/extra/completions'
    api_secret_key = "6d60510bc67941ffbc7a72ed5e945fe7"  # your api_secret_key
    #api_secret_key = "zWZiI8pCIkf7xQJ8GCnNpXCxVVihjFhz"  # your api_secret_key

    data = {
        "model": 'nova-ptc-xl-v1.2.1-0810',
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.8,
        "top_p": 0.7,
        "max_new_tokens": 512,
        "repetition_penalty": 1.05,
        "stream": False,
        "user": "test"
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': api_secret_key
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        return response.json()['data']['choices'][0]['message']
    except Exception as e:
        return f'something wrong | {response} | {e}'

def nova_8k_generate(message):
    sensenova.access_key_id = "2XqSwZTXGkXPwDkqtG6UsSvnzWT" #"2Y9AMDj6wCW8XjjxDscJm5YyUrP" 2XqSwZTXGkXPwDkqtG6UsSvnzWT
    sensenova.secret_access_key = "95MVDylccaffk6ouYKGTa8DteJLLwCgq" #"BFAe2OIluVnUCC2ujRHxE03Di7WCtINI" 95MVDylccaffk6ouYKGTa8DteJLLwCgq

    print('using nova-ptc-xl-v2-1-0-8k-internal model')
    response = sensenova.ChatCompletion.create(model = "nova-ptc-xl-v2-1-0-8k-internal",
        messages = [{"role":"user", "content":message}])

    if response:
        return response['data']['choices'][0]['message']

def nova_sensechatv5_32k_generate(message):
    # internlm2 model downloaded from huggingface
    url="https://devsft.ams-endpoint.cn-sh-01.sensecoreapi.com/internlm2-chat-1-8b-hf"
    key='eyJhbGciOiJFUzI1NiIsImtpZCI6IjBlZWRiMzg3LWMzZjktNDgzNC04ZDg5LWQ3NTc1OTkyZTIzNSIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMjUxNjQyMzQsImlhdCI6MTcwOTYzMTQzNCwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjkzMjZmNGI4LTU4MzAtNDI3MS1iN2EyLTRhMDE1NDMxOTg0NSIsInJlc291cmNlX2lkIjoiOWVhZDM3MTgtZGFkMy0xMWVlLWEwZDAtOTIwYmNlYmFlOWIzIiwic3ViIjoiOTkyN2QzNjEyZjgzMGU1ZWFmMWE1YzYzMDQ2ZTMyYjEiLCJ1cmkiOiJkZXZzZnQuYW1zLWVuZHBvaW50LmNuLXNoLTAxLnNlbnNlY29yZWFwaS5jb20vaW50ZXJubG0yLWNoYXQtMS04Yi1oZiJ9.2zAQ5Bgf_rpdZ04Jz17NxONdCeMPCdVyNV7lJiQ0PogecSihfnjFYX_ZnhX4yjYXLMMpJPVqNOQn-PgchU_7qQ'


    #print('using nova-sensechatv5-32k-testing-version model')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}'
    }

    data = {
        'inputs': message,
        'parameters': {
        'do_sample': False,
        'ignore_eos': False,
        #'max_new_tokens': 1024,
        'max_new_tokens': 2048,
        'stop_sequences': '<|im_end|>',
        #'top_k': 50
         'top_k': 50
        },
        'stream': False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    if response.status_code == 200:
        generated_text = response.json()[0]['generated_text']
        # generated_text = generated_text.split(stop_sign)[0]
        #print(f'{generated_text}\n')
        return generated_text
    else:
        print('Error:', response.status_code, response.text)
        return response.status_code



def azure_gpt35_turbo_generate(message, openai_key, openai_endpoint) -> str:

    AZURE_OPENAI_KEY = openai_key
    AZURE_OPENAI_ENDPOINT = openai_endpoint

    client = AzureOpenAI(
        #azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        #api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        api_key = AZURE_OPENAI_KEY,
        api_version="2023-05-15"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-35-turbo", # model = "deployment_name".
            messages=[{"role": "user", "content": message }],
        )
        #return response.json()['data']['choices'][0]['message']
        return response.choices[0].message.content
    except Exception as e:
        return f'something wrong | {response} | {e}'

def azure_gpt4_generate(message, openai_key, openai_endpoint) -> str:

    AZURE_OPENAI_KEY = openai_key
    AZURE_OPENAI_ENDPOINT = openai_endpoint

    client = AzureOpenAI(
        #azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        #api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        api_key = AZURE_OPENAI_KEY,
        #api_version="2023-05-15"
        api_version="2024-02-01"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4", # model = "deployment_name".
            messages=[{"role": "user", "content": message }],
        )
        #return response.json()['data']['choices'][0]['message']
        return response.choices[0].message.content
    except Exception as e:
        return f'something wrong | {response} | {e}'


def azure_gpt4_32k_generate(message, openai_key, openai_endpoint) -> str:

    AZURE_OPENAI_KEY = openai_key
    AZURE_OPENAI_ENDPOINT = openai_endpoint

    client = AzureOpenAI(
        #azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        #api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        api_key = AZURE_OPENAI_KEY,
        #api_version="2023-05-15"
        api_version="2024-02-01"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-32k", # model = "deployment_name".
            messages=[{"role": "user", "content": message }],
            #temperature = 0.8,
            #max_tokens = 4096,
            #top_p = 0.9,
            #frequency_penalty = 0,
            #presence_penalty = 0,
            #stop = None
        )
        #return response.json()['data']['choices'][0]['message']
        return response.choices[0].message.content
    except Exception as e:
        return f'something wrong | {response} | {e}'

def gpt4o_ptu_generate(message,openai_key, openai_endpoint=None) -> str:
    #import pdb;pdb.set_trace()
    client = openai_proxy.GptProxy(api_key=openai_key)
    succeed = False
    while not succeed:
        try:
            response = client.generate(
                messages=[{"role": "user","content": message}],
                model="gpt-4o-2024-08-06",
                transaction_id="senserag_benchmark", # 同样transaction_id将被归类到同一个任务，一起统计
            )
            succeed=True
            return response.json()['data']['response_content']['choices'][0]['message']['content']
        except Exception as e:
            print(f'something wrong | {response.text} | {e}')
            print('retry again')


def remote_generate(model_name: str) -> Callable:
    model2func = dict(
        chatgpt=api2d_generate,
        #euclid=euclid_generate,
        nova_8k=nova_8k_generate,
        nova_sensechatv5=nova_sensechatv5_32k_generate,
        azure_gpt35_turbo=azure_gpt35_turbo_generate,
        azure_gpt4=azure_gpt4_generate,
        azure_gpt4_32k=azure_gpt4_32k_generate,
        nova=nova_v2_generate,
        gpt4o_ptu=gpt4o_ptu_generate
    )
    return model2func[model_name]
