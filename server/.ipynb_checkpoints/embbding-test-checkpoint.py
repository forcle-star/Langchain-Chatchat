import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

# 这里替换成你自己的词向量模型文件夹路径
local_model_dir = "/root/model/bge-large-zh"

# 如果没有显卡，将下面这句话注释
model_kwargs = {"device": "cuda:0"}

# 是否需要将模型输出再做一次归一化。对于bge模型来说，输出本身就是单位向量，不需要。
encode_kwargs = {"normalize_embeddings": False}
print(f"loading embedding model {local_model_dir}...")
hf_embedding_model = HuggingFaceEmbeddings(
    model_name=local_model_dir,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
print(f"embedding model {local_model_dir} loaded")
text1 = "我正在使用大模型帮助我写作"
text2 = "我经常使用大模型"
text3 = "我爱骑自行车"


print(f"验证{local_model_dir}词向量模型生成的向量长度为1，是单位向量:")
vector1 = hf_embedding_model.embed_query(text1)
print(text1, np.linalg.norm(vector1))
vector2 = hf_embedding_model.embed_query(text2)
print(text2, np.linalg.norm(vector2))
vector3 = hf_embedding_model.embed_query(text3)
print(text3, np.linalg.norm(vector3))

from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.utils import DistanceStrategy

# cos(a, b) = inner_product(a, b) / (||a|| * ||b||)
# 已知向量长度为1，所以只需要考虑分子：也就是内积。
# 指定distance_strategy入参，会在 python3.10/site-packages/langchain/vectorstores/faiss.py 515 行产生区别。langchain 0.0.250版本。
# 其他langchain版本自行从from_documents点进去看源码。
from langchain.schema import Document

doc1 = Document(page_content=text1, metadata={"xxxx": "可以放一些出处信息，比如文章id，或者原始记录中的id", "id":1})
doc2 = Document(page_content=text2, metadata={"xxxx": "可以放一些出处信息，比如文章id，或者原始记录中的id", "id":2})
doc3 = Document(page_content=text3, metadata={"xxxx": "可以放一些出处信息，比如文章id，或者原始记录中的id", "id":3})
faiss_index = FAISS.from_documents([doc1, doc2, doc3], embedding = hf_embedding_model, distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT)
faiss_index: FAISS

# 用 text1, text2, text3 分别搜索三次
def print_result_with_score(query, result):
    print("="*20)
    print("搜索:", query)
    for doc, score in result:
        print(score, doc)

result = faiss_index.similarity_search_with_score(text1)
print_result_with_score(text1, result)
result = faiss_index.similarity_search_with_score(text2)
print_result_with_score(text2, result)
result = faiss_index.similarity_search_with_score(text3)
print_result_with_score(text3, result)