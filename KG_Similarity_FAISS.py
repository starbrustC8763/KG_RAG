from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import faiss
import time
import os
load_dotenv()
# 使用環境變數
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 索引保存路徑
INDEX_PATH = "fact_index.faiss"

# 構建 FAISS 索引（如果不存在）
def build_faiss_index():
    print("Building FAISS index from Neo4j data...")
    with driver.session() as session:
        results = session.run("MATCH (f:Fact) RETURN f.id AS id, f.text AS text, f.embedding AS embedding")
        embeddings = []
        fact_ids = []
        fact_texts = []
        
        for record in results:
            fact_ids.append(record["id"])
            fact_texts.append(record["text"])
            embeddings.append(np.array(record["embedding"], dtype="float32"))

    # 構建 FAISS 索引
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)  # 使用 L2 距離
    index.add(np.array(embeddings))  # 添加向量
    print(f"Built FAISS index with {len(embeddings)} entries.")
    
    # 保存索引和元數據
    faiss.write_index(index, INDEX_PATH)
    with open("fact_metadata.npy", "wb") as f:
        np.save(f, {"fact_ids": fact_ids, "fact_texts": fact_texts})
    
    return index, fact_ids, fact_texts

# 初始化嵌入模型
model = SentenceTransformer("shibing624/text2vec-base-chinese")
# 加載 FAISS 索引（如果存在）
def load_faiss_index():
    if os.path.exists(INDEX_PATH) and os.path.exists("fact_metadata.npy"):
        print("Loading FAISS index from disk...")
        index = faiss.read_index(INDEX_PATH)
        metadata = np.load("fact_metadata.npy", allow_pickle=True).item()
        return index, metadata["fact_ids"], metadata["fact_texts"]
    else:
        return build_faiss_index()

# 初始化嵌入模型
model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 加載或構建 FAISS 索引
index, fact_ids, fact_texts = load_faiss_index()
# 查詢最近鄰
def query_faiss(input_text, model, top_k=5):
    # 生成查詢向量
    query_embedding = np.array([model.encode(input_text)], dtype="float32")
    
    # 查詢最近鄰
    distances, indices = index.search(query_embedding, top_k)
    results = []
    
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "id": fact_ids[idx],
            "text": fact_texts[idx],
            "distance": dist
        })
    return results

# 函數：查詢每個案件所引用的法條
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

# 測試輸入數據
input_fact = """
一、事故發生緣由:
被告劉祥明受僱於新環遊交通股份有限公司，於民國110年10月29日下午5時27分左右，駕駛車牌號碼000-0000號營業用大客車，在國道3號高速公路外側車道由南往北行駛。行經12公里400公尺處時，被告應該要注意與前車保持安全距離，但是他沒有注意到前方車輛已經減慢速度，也沒有及時保持安全距離，結果從後方追撞了原告駕駛的車牌號碼0000-00號自用小客車的車尾。

二、原告受傷情形:
原告因為這次車禍受到腦震盪、頸部拉傷、胸部背部髖部挫傷等傷害。原告在三軍總醫院急診後，醫生建議休養3天。之後原告持續在亞東紀念醫院神經外科門診治療，醫生多次建議在家休養，包括2星期、1個月等。直到111年5月6日的門診，醫生仍然表示原告目前無法進行工作。整體來說，原告因為這次車禍，從受傷到111年5月底，總共有7個月的時間無法工作。
"""

# 查詢最相似的事實
start_time = time.time()
results = query_faiss(input_fact, model)

# 打印結果
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}, Text: {result['text']}")
    # 查詢該案件引用的法條
    statutes_info = get_statutes_for_case(result["id"])
    for info in statutes_info:
        print(f"引用的法條: {', '.join(info['statutes'])}")

end_time = time.time()
print(f"執行時間: {end_time - start_time} 秒")
