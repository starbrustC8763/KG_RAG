from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import numpy as np
import torch
import time  # 導入 time 模組
start_time = time.time()
# 加載 .env 文件中的環境變數
load_dotenv()

# 使用環境變數
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 初始化嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 函數：從 Neo4j 獲取嵌入並計算相似度
def get_similar_facts(input_text, top_k=10):
    input_embedding = model.encode(input_text)
    input_embedding = torch.tensor(input_embedding, dtype=torch.float32).to(device)

    with driver.session() as session:
        # 獲取所有節點的嵌入向量
        results = session.run("MATCH (f:Fact) RETURN f.id AS id, f.text AS text, f.embedding AS embedding")
        fact_ids = []
        fact_texts = []
        embeddings = []

        for record in results:
            fact_ids.append(record["id"])
            fact_texts.append(record["text"])
            # 將嵌入向量轉換為 float32 類型
            embeddings.append(np.array(record["embedding"], dtype=np.float32))

        # 將嵌入轉換為 PyTorch 張量並移動到 GPU
        embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32).to(device)

        # 計算餘弦相似度
        input_norm = input_embedding / input_embedding.norm()
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        similarities = torch.matmul(embeddings_norm, input_norm)

        # 將結果轉回 CPU 並排序
        similarities = similarities.cpu().numpy()
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_results = [(fact_ids[i], fact_texts[i], similarities[i]) for i in top_indices]

    return top_results

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

# 示例：輸入嵌入向量，找到最相似的案件事實
input_fact = """
一、事故發生緣由:
被告劉祥明受僱於新環遊交通股份有限公司，於民國110年10月29日下午5時27分左右，駕駛車牌號碼000-0000號營業用大客車，在國道3號高速公路外側車道由南往北行駛。行經12公里400公尺處時，被告應該要注意與前車保持安全距離，但是他沒有注意到前方車輛已經減慢速度，也沒有及時保持安全距離，結果從後方追撞了原告駕駛的車牌號碼0000-00號自用小客車的車尾。

二、原告受傷情形:
原告因為這次車禍受到腦震盪、頸部拉傷、胸部背部髖部挫傷等傷害。原告在三軍總醫院急診後，醫生建議休養3天。之後原告持續在亞東紀念醫院神經外科門診治療，醫生多次建議在家休養，包括2星期、1個月等。直到111年5月6日的門診，醫生仍然表示原告目前無法進行工作。整體來說，原告因為這次車禍，從受傷到111年5月底，總共有7個月的時間無法工作。
"""
similar_facts = get_similar_facts(input_fact)

# 輸出結果
i=1
for fact_id, fact_text, similarity in similar_facts:
    print(f"第{i}相似")
    print(f"事實 ID: {fact_id}")
    print(f"事實內容: {fact_text}")
    print(f"相似度: {similarity}")
    i+=1
    # 查詢該案件引用的法條
    statutes_info = get_statutes_for_case(fact_id)
    for info in statutes_info:
        print(f"所屬案件 ID: {info['case_id']}")
        print(f"引用的法條: {', '.join(info['statutes'])}")
    print()

end_time = time.time()
print(f"執行時間: {end_time - start_time} 秒")