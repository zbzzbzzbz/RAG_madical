from typing_extensions import List
import os


PROMPT_TEMPLATE = dict(
    RAG_SYSTEM_TEMPALTE="""
        你是一个针对医疗档案进行问答的机器人。你的任务是根据下述给定的已知信息回答用户问题。请不要输出已知信息中不包含的信息或答案。

        【任务设定】  
        1. **请严格参考知识库或者上下文来回答用户的问题**：如果你不知道答案，就说你不知道。
        - 如果给定的知识库无法让你做出回答，请参考上下文，若知识库和上下文都无法让你做出回答，就说你不知道, 不要擅自回答。
        2. **总是使用中文回答**
        3. **格式要求**:
        - 控制回答token数在150以内, 尽量使回答内容精简

        可参考的知识库：{context}

        如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
        请不要输出已知信息中不包含的信息或答案。
        请不要输出已知信息中不包含的信息或答案。
        请不要输出已知信息中不包含的信息或答案。
        """,
    
    RAG_PROMPT_TEMPALTE="""
        问题: {question}
        """,

    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError
    


class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo") -> None:
        super().__init__(path)
        from openai import OpenAI
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")   
        self.client.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        systemmeg = [{"role": "system", "content":PROMPT_TEMPLATE['RAG_SYSTEM_TEMPALTE'].format(context=content)}]
        messages = systemmeg + history + [{"role": "user", "content": PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt)}]
        print(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content


class ZhipuChat(BaseModel):
    def __init__(self, path: str = '', model: str = "glm-4") -> None:
        super().__init__(path)
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        systemmeg = [{"role": "system", "content":PROMPT_TEMPLATE['RAG_SYSTEM_TEMPALTE'].format(context=content)}]
        messages = systemmeg + history + [{"role": "user", "content": PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt)}]
        print(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content