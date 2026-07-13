#!/bin/bash
python ./auto_eval_for_rag.py --GPT_iter 2 --openai_key XXXXXXX --openai_endpoint https://XXXXXXXX.openai.azure.com/ --read_jsonl_file example_eval.jsonl --write_jsonl_file rating_example_eval.jsonl
