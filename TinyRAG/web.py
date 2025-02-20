'''
    运用Streamlit完成前端页面显示
'''
import streamlit as st
from VectorBase import VectorStore
from Files import ReadFiles
from LLM import OpenAIChat, ZhipuChat
from Embeddings import M3eBaseEmbedding, BaseEmbeddings
from typing_extensions import List


st.title("💬 RAG大模型")
st.caption("🚀 聊天机器人developed by zbz")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，有什么可以帮您?"}]


# 更新向量库
@st.cache_resource
def updateVector(_embedding: BaseEmbeddings):
    docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
    print("ReadFiles done \n")
    vector = VectorStore(document = docs)
    print("vector = VectorStore(document = docs) done \n")
    vector.get_vector(EmbeddingModel = _embedding)
    print("vector.get_vector done \n")
    vector.persist(path='C:/Users/zheng/Desktop/bishe/langchain_learning/VecDocData')
    print("vector.persist done \n")
    return vector


# 初始化模型
def init_models():
    # 加载embedding model
    embedding = M3eBaseEmbedding()
    print("embedding = M3eBaseEmbedding() done \n")
    st.session_state["Embedding_Model"] = embedding
    # 加载chat model
    chat = ZhipuChat()
    print("chat = ZhipuChat() done \n")
    st.session_state["Chat_Model"] = chat
    # 加载现有向量库
    vector = VectorStore()
    print("vector = VectorStore() done \n")
    vector.load_vector(path='C:/Users/zheng/Desktop/bishe/langchain_learning/VecDocData')
    print("load_vector done \n")
    st.session_state["vector"] = vector 


# 检查是否需要初始化模型
if "Embedding_Model" not in st.session_state:
    with st.spinner("加载模型ing..."):
        init_models()
        print("init_models() done")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "您好，有什么可以帮您?"}]
st.sidebar.button('清空聊天记录', on_click=clear_chat_history)

def updateVector():
    with st.spinner("更新向量库ing..."):
        if "Embedding_Model" in st.session_state:
            docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
            print("ReadFiles done \n")
            vector = VectorStore(document = docs)
            print("vector = VectorStore(document = docs) done \n")
            vector.get_vector(EmbeddingModel = st.session_state["Embedding_Model"])
            print("vector.get_vector done \n")
            vector.persist(path='C:/Users/zheng/Desktop/bishe/langchain_learning/VecDocData')
            print("vector.persist done \n")
            st.session_state["vector"] = vector
st.sidebar.button('更新向量库', on_click=updateVector)


# 显示对话历史
for message in st.session_state["messages"]:
    if message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
    else:
        st.chat_message("user").write(message["content"])

# 获取用户输入
user_input = st.chat_input()
if user_input:
    st.chat_message("user").write(user_input)
    contents = st.session_state["vector"].query(
        user_input, 
        EmbeddingModel=st.session_state["Embedding_Model"], 
        k=2
    )
    content = "\n".join(contents)
    response = st.session_state["Chat_Model"].chat(user_input, st.session_state["messages"], content)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": response})
    print(f"response : {response}")
    st.rerun()