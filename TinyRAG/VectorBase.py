from typing_extensions import List
from Embeddings import BaseEmbeddings
import numpy as np
from tqdm import tqdm
import os
import json


class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document
    
    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # 获得文档的向量表示
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            # 对 self.document 列表进行迭代，并显示一个进度条
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors


    def persist(self, path: str = 'storage'):
        # 数据库持久化，本地保存
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", 'w', encoding='utf-8') as file:
            # ensure_ascii=False 参数表示在写入 JSON 数据时，不将非 ASCII 字符转义
            json.dump(self.document, file, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as file:
                json.dump(self.vectors, file)
        

    def load_vector(self, path: str = 'storage'):
        # 从本地加载数据库
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as file:
            self.vectors = json.load(file)
        with open(f"{path}/document.json", 'r', encoding='utf-8') as file:
            self.document = json.load(file)
        

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)


    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # 根据问题检索相关的文档片段
        query_vector = EmbeddingModel.get_embedding(query)
        # 列表转数组，便于高效计算
        result = np.array([self.get_similarity(query_vector, vector) for vector in self.vectors])
        # 返回相似度最高的k个文档
        # .argsort() 返回从小到大排序后的索引数组
        # [::-1] 反转数组
        # .tolist() 数组转列表
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
    


