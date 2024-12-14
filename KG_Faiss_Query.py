from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from dotenv import load_dotenv
load_dotenv()

# Neo4j 配置
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 索引保存路徑
INDEX_PATH = "fact_index_hnsw.faiss"

# 初始化嵌入模型
model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 構建 FAISS 索引
def build_faiss_index():
    with driver.session() as session:
        results = session.run("MATCH (f:Fact) RETURN f.id AS id, f.text AS text, f.embedding AS embedding")
        embeddings = []
        fact_ids = []
        fact_texts = []
        
        for record in results:
            fact_ids.append(record["id"])
            fact_texts.append(record["text"])
            embeddings.append(np.array(record["embedding"], dtype="float32"))

    dimension = len(embeddings[0])
    M = 32
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100
    index.add(np.array(embeddings))
    
    faiss.write_index(index, INDEX_PATH)
    with open("fact_metadata_hnsw.npy", "wb") as f:
        np.save(f, {"fact_ids": fact_ids, "fact_texts": fact_texts})
    
    return index, fact_ids, fact_texts

# 加載 FAISS 索引
def load_faiss_index():
    if os.path.exists(INDEX_PATH) and os.path.exists("fact_metadata_hnsw.npy"):
        index = faiss.read_index(INDEX_PATH)
        metadata = np.load("fact_metadata_hnsw.npy", allow_pickle=True).item()
        return index, metadata["fact_ids"], metadata["fact_texts"]
    else:
        return build_faiss_index()

# 查詢 FAISS
def query_faiss(input_text, top_k=5):
    query_embedding = np.array([model.encode(input_text)], dtype="float32")
    index, fact_ids, fact_texts = load_faiss_index()
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "id": fact_ids[idx],
            "text": fact_texts[idx],
            "distance": dist
        })
    return results

# 查詢引用法條
def get_statutes_for_case(fact_id):
    with driver.session() as session:
        results = session.run(
            """
            MATCH (c:Case)-[:案件事實]->(f:Fact {id: $fact_id})
            MATCH (c)-[:案件相關法條]->(l:LegalReference)
            MATCH (l)-[:引用法條]->(s:Statute)
            RETURN c.id AS case_id, collect(s.id) AS statutes
            """,
            fact_id=fact_id
        )
        return [{"case_id": record["case_id"], "statutes": record["statutes"]} for record in results]
