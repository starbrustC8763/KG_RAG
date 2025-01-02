from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
# 連接到 Neo4j
# 加載 .env 文件中的環境變數
load_dotenv()

# 使用環境變數
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 加載嵌入模型
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 提取節點文本並生成嵌入向量
def add_embeddings_to_nodes():
    with driver.session() as session:
        # 提取所有需要嵌入的節點
        nodes = session.run("MATCH (n) RETURN elementId(n) AS id, n.text AS text")
        
        for record in nodes:
            node_id = record["id"]
            text = record["text"]
            if text:  # 確保文本不為空
                # 生成嵌入向量
                embedding = model.encode(text).tolist()
                # 更新節點，將嵌入向量存為屬性
                session.run(
                    "MATCH (n) WHERE elementId(n) = $id SET n.embedding = $embedding",
                    id=node_id, embedding=embedding
                )

# 執行嵌入添加
add_embeddings_to_nodes()
