from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv

# 加載 .env 配置
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

def build_faiss_index() -> Tuple[faiss.IndexHNSWFlat, List[str], List[str]]:
    """
    從 Neo4j 數據庫中構建 FAISS 索引並保存到磁盤。

    Returns:
        Tuple[faiss.IndexHNSWFlat, List[str], List[str]]: FAISS 索引，事實節點 ID 列表，事實文本列表。
    """
    with driver.session() as session:
        # 查詢所有 Fact 節點的 ID、文本和嵌入
        results = session.run("MATCH (f:Fact) RETURN f.id AS id, f.text AS text, f.embedding AS embedding")
        embeddings = []
        fact_ids = []
        fact_texts = []
        
        for record in results:
            fact_ids.append(record["id"])
            fact_texts.append(record["text"])
            embeddings.append(np.array(record["embedding"], dtype="float32"))

    # 構建 FAISS HNSW 索引
    dimension = len(embeddings[0])
    M = 32  # HNSW 的參數，決定連接數量
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = 200  # 構建時的 ef 值
    index.hnsw.efSearch = 100  # 查詢時的 ef 值
    index.add(np.array(embeddings))  # 添加嵌入向量

    # 保存索引到磁盤
    faiss.write_index(index, INDEX_PATH)
    with open("fact_metadata_hnsw.npy", "wb") as f:
        np.save(f, {"fact_ids": fact_ids, "fact_texts": fact_texts})
    
    return index, fact_ids, fact_texts

def load_faiss_index() -> Tuple[faiss.IndexHNSWFlat, List[str], List[str]]:
    """
    加載 FAISS 索引和對應的元數據。如果索引不存在，則構建索引。

    Returns:
        Tuple[faiss.IndexHNSWFlat, List[str], List[str]]: FAISS 索引，事實節點 ID 列表，事實文本列表。
    """
    if os.path.exists(INDEX_PATH) and os.path.exists("fact_metadata_hnsw.npy"):
        # 從磁盤加載索引
        index = faiss.read_index(INDEX_PATH)
        metadata = np.load("fact_metadata_hnsw.npy", allow_pickle=True).item()
        return index, metadata["fact_ids"], metadata["fact_texts"]
    else:
        # 索引不存在時構建索引
        return build_faiss_index()

def query_faiss(input_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    在 FAISS 索引中查詢最相似的事實。

    Args:
        input_text (str): 用戶輸入的文本。
        top_k (int): 返回的最相似事實數量。

    Returns:
        List[Dict[str, Any]]: 包含最相似事實的 ID、文本和距離的列表。
    """
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

def get_statutes_for_case(fact_id: str) -> List[Dict[str, Any]]:
    """
    查詢指定事實所屬案件引用的法條。

    Args:
        fact_id (str): 事實節點的 ID。

    Returns:
        List[Dict[str, Any]]: 包含案件 ID 和引用法條的列表。
    """
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

def fetch_statutes_and_explanations(statutes: List[str]) -> List[Dict[str, str]]:
    """
    查詢指定法條的條文內容及其口語化解釋。

    Args:
        statutes (List[str]): 要查詢的法條 ID 列表。

    Returns:
        List[Dict[str, str]]: 包含法條 ID、條文和口語化解釋的字典列表。
    """
    query = """
    MATCH (s:Statute)-[:口語化解釋]->(e:Explanation)
    WHERE s.id IN $statutes
    RETURN s.id AS statute_id, s.text AS statute_text, e.text AS explanation_text
    """
    with driver.session() as session:
        results = session.run(query, statutes=statutes)
        return [
            {
                "statute_id": record["statute_id"],
                "statute_text": record["statute_text"],
                "explanation_text": record["explanation_text"]
            }
            for record in results
        ]

def get_legal(case_facts: str, injury_details: str) -> str:
    """
    根據案件事實和受傷情形生成相關的法條引用。

    Args:
        case_facts (str): 案件事實。
        injury_details (str): 受傷情形。

    Returns:
        str: 相關法條的字符串列表。
    """
    input_text = f"{case_facts} {injury_details}"
    similar_facts = query_faiss(input_text, top_k=5)
    statutes_set = set()

    for fact in similar_facts:
        fact_id = fact["id"]
        statutes_info = get_statutes_for_case(fact_id)
        for info in statutes_info:
            statutes_set.update(info["statutes"])

    legal_references = "\n".join(sorted(statutes_set))
    return legal_references
