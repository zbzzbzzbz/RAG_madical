import faiss
import numpy as np

# 向量维度
d = 64  
# 数据集中向量的数量
nb = 1000  
# 随机生成数据集
xb = np.random.random((nb, d)).astype('float32')  

# 创建基于L2距离的平面索引
index = faiss.IndexFlatL2(d)
# 将向量数据添加到索引中
index.add(xb)  

# 查询向量的数量
nq = 1  
# 随机生成查询向量
xq = np.random.random((nq, d)).astype('float32')  

# 搜索与查询向量最相似的k个向量
k = 4  
distances, indices = index.search(xq, k)

print("查询向量与最相似向量的距离:", distances)
print("最相似向量在数据集中的索引:", indices)