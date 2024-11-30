from neo4j import GraphDatabase
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import time  # 導入 time 模組
start_time = time.time()
# 加載 .env 文件中的環境變數
load_dotenv()
# 使用環境變數
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))
pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
def query_pinecone(input_text, model, index_name="fact-embeddings-300", top_k=5):
    # 初始化嵌入模型（如未初始化）
    embedding_model = model if model else SentenceTransformer("shibing624/text2vec-base-chinese")

    # 生成輸入文本的嵌入
    query_vector = embedding_model.encode(input_text).tolist()

    index = pc.Index(index_name)

    # 查詢 Pinecone
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # 提取結果
    results = [
        {
            "id": match.id,
            "text": match.metadata["text"],
            "score": match.score
        }
        for match in response.matches
    ]

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

# 初始化嵌入模型
model = SentenceTransformer("shibing624/text2vec-base-chinese")
input_fact = """
一、事故發生緣由:
被告劉祥明受僱於新環遊交通股份有限公司，於民國110年10月29日下午5時27分左右，駕駛車牌號碼000-0000號營業用大客車，在國道3號高速公路外側車道由南往北行駛。行經12公里400公尺處時，被告應該要注意與前車保持安全距離，但是他沒有注意到前方車輛已經減慢速度，也沒有及時保持安全距離，結果從後方追撞了原告駕駛的車牌號碼0000-00號自用小客車的車尾。

二、原告受傷情形:
原告因為這次車禍受到腦震盪、頸部拉傷、胸部背部髖部挫傷等傷害。原告在三軍總醫院急診後，醫生建議休養3天。之後原告持續在亞東紀念醫院神經外科門診治療，醫生多次建議在家休養，包括2星期、1個月等。直到111年5月6日的門診，醫生仍然表示原告目前無法進行工作。整體來說，原告因為這次車禍，從受傷到111年5月底，總共有7個月的時間無法工作。
"""
# 查詢最相似的事實
results = query_pinecone(input_fact, model)

# 打印結果
for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}, Text: {result['text']}")
    # 查詢該案件引用的法條
    statutes_info = get_statutes_for_case(result["id"])
    for info in statutes_info:
        print(f"引用的法條: {', '.join(info['statutes'])}")
end_time = time.time()     
print(f"執行時間: {end_time - start_time} 秒")   