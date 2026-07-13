# coding:utf-8
import json
import logging
import time
from logging.handlers import TimedRotatingFileHandler

import requests
import sensenova
import os
from functools import wraps


EMBEDDING_PATH = "/home/pretrained_models/m3e-base"
# dense embedding是否使用norm
EMBEDDING_NORM=True
#EMBEDDING_NORM=False

# 选择哪个大语言模型
# 默认用GPT4-32K
#model_name = "nova"

#True: Do not call GPT API when cos_sim >= 0.96
COS_THRESHOLD = False

#the account of GPT API Call for each question
GPT_ITERATION = 1
