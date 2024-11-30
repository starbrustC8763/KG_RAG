from neo4j import GraphDatabase
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
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
# 提取所有 Fact 節點數據
def fetch_all_facts():
    with driver.session() as session:
        query = """
        MATCH (f:Fact)
        RETURN f.id AS id, f.text AS text, f.embedding AS embedding
        """
        result = session.run(query)
        facts = [
            {
                "id": record["id"],
                "text": record["text"],
                "embedding": record["embedding"]
            }
            for record in result
        ]
    return facts
# 轉換為 Pinecone 格式
def transform_to_pinecone_format(facts):
    return [
        (
            fact["id"],
            fact["embedding"],
            {"text": fact["text"]}
        )
        for fact in facts
    ]

# 插入數據到 Pinecone
def insert_to_pinecone(data):
    index_name = "fact-embeddings-300"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # 嵌入向量的維度
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index(index_name)
    index.upsert(vectors=data)

# 主程序
if __name__ == "__main__":
    # 提取數據並插入 Pinecone
    facts = fetch_all_facts()
    pinecone_data = transform_to_pinecone_format(facts)
    insert_to_pinecone(pinecone_data)