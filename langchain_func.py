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
# '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'


# q = str("안녕하세요 또 막상 이렇게 하니까 또 뭘 말하기가 좀 애매하긴 한데  그냥 막 말해봐요 어차피 두 사람이라고 마이크는 꼭 하자 가까이 두세요 얘도 이제 그런 문제가 있는 거잖아요 그쵸? 중요한 순간에는 굿마크를 탭하세요 아 조금 중요하다 생각되는 거는  이걸 누르면 거기를 좀 집중적으로 해 주나 봐요 전처리를 뭐 새로 따로 해 주나 보네  근데 이 정도 거리에서 말했을 때  마이크 반 칸 정도 차는 거 보니까 좀 잘 안 들리는 걸 수도 있죠 그 약간 이런 전체적인 대화라기보다는 그냥 근데 조금은 다른 게 전에는 뭐 하자 이렇게 몇 명 지정도 하고 그랬는데 이건 그냥 녹음하는 건가? 아니면 그냥 녹음 방식으로 바꾼 것일 수도 있을 거 같아 여기서도 녹음 방식으로요? 네 여기서 녹음으로 한다고 그랬었어요 전에도 그랬나? 그러니까 아마 이 버튼을 누르는 순간에는 그때 저장되는 그 밴드가 넓어지겠죠 그런 식으로 해가지고 전처리가 따로 들어가는 거 같은데 그럴 수도 있겠지")

# print(llm.predict("안녕!"))
