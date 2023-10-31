# -*- coding: utf-8 -*-
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=api_key)

llm = OpenAI()
chat_model = ChatOpenAI()
def summary(q):
    summary = """ 질문 내용에 의거하여 질문의 핵심만 간결하게 요약해줘 . 
    질문: {question}"""
    prompt = PromptTemplate.from_template(summary)
    ans = chat_model.predict(prompt.format(question=q), temperature=1)
    return ans
def word_extraction(q):
    word_extraction = """질문에서 핵심 단어를 최대 3가지 추출하시오. 
    질문: {question}"""
    summary_prompt = PromptTemplate.from_template(word_extraction)
    ans = chat_model.predict(summary_prompt.format(question=q), temperature=1)
    return ans
