from typing_extensions import List
import os
import numpy as np

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    

class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            # 多行文本转为单行文本，利于大模型分析处理
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError


class M3eBaseEmbedding(BaseEmbeddings):
    """
    class for M3e-Base embeddings
    """
    def __init__(self, path: str = 'EmbeddingModels/m3e-base', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        if not self.is_api:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(path)

    def get_embedding(self, text: str, model: str = '') -> List[float]:
        if not self.is_api:
            # 多行文本转为单行文本，利于大模型分析处理
            text = text.replace("\n", " ")
            return self.model.encode(text).tolist()
        else:
            raise NotImplementedError


