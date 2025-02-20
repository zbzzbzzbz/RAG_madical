from VectorBase import VectorStore
from Files import ReadFiles
from LLM import OpenAIChat, ZhipuChat
from Embeddings import M3eBaseEmbedding
from typing_extensions import List

class Rag:
    def __init__(self, question):
        self.question = question

    def loadEmbeddingModel():
        embedding = M3eBaseEmbedding()
        return embedding

    def getResponse(self, history: List[dict]):
        print(f"getResponse start, question : {self.question} \n")
        if self.question:
            # 获得data目录下的所有文件内容并分割
            vector = VectorStore()
            print("vector = VectorStore() done \n")
            embedding = M3eBaseEmbedding()
            print("embedding = M3eBaseEmbedding() done \n")
            vector.load_vector(path='C:/Users/zheng/Desktop/bishe/langchain_learning/VecDocData')
            print("load_vector done \n")
            content = vector.query(self.question, EmbeddingModel=embedding, k=1)[0]
            print("vector.query done \n")
            # chat = OpenAIChat()
            chat = ZhipuChat()
            return chat.chat(self.question, history, content)
        
    def testResponse(self, history: List[dict]):
        print(f"testResponse start, question : {self.question} \n")
        if self.question:
            docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
            vector = VectorStore(docs)
            print("vector = VectorStore() done \n")
            embedding = M3eBaseEmbedding()
            print("embedding = M3eBaseEmbedding() done \n")
            vector.get_vector(EmbeddingModel=embedding)
            print("get_vector done \n")
            vector.persist(path='C:/Users/zheng/Desktop/bishe/langchain_learning/VecDocData')
            print("persist done \n")
            content = vector.query(self.question, EmbeddingModel=embedding, k=1)[0]
            print("vector.query done \n")
            # chat = OpenAIChat()
            chat = ZhipuChat()
            return chat.chat(self.question, history, content)

