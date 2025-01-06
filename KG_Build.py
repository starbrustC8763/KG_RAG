from neo4j import GraphDatabase
from dotenv import load_dotenv
import re
import os
# 加載 .env 文件中的環境變數
load_dotenv()
# 連接到 Neo4j 資料庫
# 使用環境變數
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 函數：創建 Statute 和 Explanation 節點
def create_statute_and_explanation(tx, statute_id, statute_text, explanation_text):
    tx.run("MERGE (s:Statute {id: $id, text: $text})", id=statute_id, text=statute_text)
    tx.run("MERGE (e:Explanation {id: $id, text: $text})", id=f"{statute_id}_explanation", text=explanation_text)
    tx.run("MATCH (s:Statute {id: $id}), (e:Explanation {id: $explanation_id}) MERGE (s)-[:口語化解釋]->(e)",
           id=statute_id, explanation_id=f"{statute_id}_explanation")
    print(f"已建立{statute_id}節點")

# 函數：創建 "起訴書相關法條" 節點
def create_law_node(tx):
    tx.run("MERGE (n:LawNode {name: '起訴書相關法條'})")
    print(f"已建立(起訴書相關法條)節點")

# 函數：將所有 "Statute" 節點連接到 "起訴書相關法條"
def link_statutes(tx):
    tx.run(
        "MATCH (s:Statute), (m:LawNode {name: '起訴書相關法條'}) "
        "MERGE (m)-[:相關法條]->(s)"
    )
    print(f"已將所有 Statute 節點連接到 起訴書相關法條")

# 函數：創建 Case 節點
def create_case_node(tx, case_id, case_text):
    tx.run("MERGE (c:Case {id: $id, text: $text})", id=case_id, text=case_text)
    print(f"已建立{case_id}節點")

# 函數：創建 Fact 節點
def create_fact_node(tx, fact_id, fact_text):
    tx.run("MERGE (f:Fact {id: $id, text: $text})", id=fact_id, text=fact_text)
    print(f"已建立{fact_id}節點")

# 函數：連接 Fact 節點到 Case 節點
def link_fact_to_case(tx, case_id, fact_id):
    tx.run("MATCH (c:Case {id: $case_id}), (f:Fact {id: $fact_id}) MERGE (c)-[:案件事實]->(f)",
           case_id=case_id, fact_id=fact_id)
    print(f"將{fact_id}連結到{case_id}")

# 函數：創建 LegalReference 節點
def create_legal_node(tx, legal_id, legal_text):
    tx.run("MERGE (l:LegalReference {id: $id, text: $text})", id=legal_id, text=legal_text)
    print(f"已建立{legal_id}節點")

# 函數：將法條格式標準化
def normalize_statute_reference(reference):
    # 將 "第191條之2" 轉換為 "191-2條"
    normalized = re.sub(r"條之(\d+)", r"-\1條", reference)
    return normalized

def create_and_link_legal_node(tx, legal_id, legal_text):
    # 創建 LegalReference 節點
    tx.run("MERGE (l:LegalReference {id: $id, text: $text})", id=legal_id, text=legal_text)
    print(f"已建立{legal_id}節點")

    # 找出所有引用的法條
    references = re.findall(r"第(\d+-?\d*條之?\d*)", legal_text)
    for ref in references:
        # 標準化引用格式
        normalized_ref = normalize_statute_reference(ref)
        statute_id = f"民法第{normalized_ref}"

        # 創建 LegalReference 節點與 Statute 節點的關係
        tx.run(
            "MATCH (l:LegalReference {id: $legal_id}), (s:Statute {id: $statute_id}) "
            "MERGE (l)-[:引用法條]->(s)",
            legal_id=legal_id, statute_id=statute_id
        )
        print(f"將{legal_id}連結到{statute_id}")

# 函數：連接 LegalReference 節點到 Case 節點
def link_legal_to_case(tx, case_id, legal_id):
    tx.run("MATCH (c:Case {id: $case_id}), (l:LegalReference {id: $legal_id}) MERGE (c)-[:案件相關法條]->(l)",
           case_id=case_id, legal_id=legal_id)
    print(f"將{legal_id}連結到{case_id}")

# 函數：創建 Compensation 節點
def create_compensation_node(tx, comp_id, comp_text):
    tx.run("MERGE (comp:Compensation {id: $id, text: $text})", id=comp_id, text=comp_text)
    print(f"已建立{comp_id}節點")

# 函數：連接 Compensation 節點到 Case 節點
def link_compensation_to_case(tx, case_id, comp_id):
    tx.run("MATCH (c:Case {id: $case_id}), (comp:Compensation {id: $comp_id}) MERGE (c)-[:賠償]->(comp)",
           case_id=case_id, comp_id=comp_id)
    print(f"將{comp_id}連結到{case_id}")

# 函數：創建 CompensationItem 節點
def create_comp_item_node(tx, item_id, item_text):
    tx.run("MERGE (item:CompensationItem {id: $id, text: $text})", id=item_id, text=item_text)
    print(f"已建立{item_id}節點")

# 函數：連接 CompensationItem 節點到 Compensation 節點
def link_comp_item_to_comp(tx, comp_id, item_id):
    tx.run("MATCH (comp:Compensation {id: $comp_id}), (item:CompensationItem {id: $item_id}) MERGE (comp)-[:細項]->(item)",
           comp_id=comp_id, item_id=item_id)
    print(f"將{item_id}連結到{comp_id}")

# 函數：創建 "參考用判決書" 節點
def create_reference_node(tx):
    tx.run("MERGE (r:ReferenceNode {name: '參考用判決書'})")
    print(f"創建 參考用判決書 節點")

# 函數：連接 Case 節點到 "參考用判決書" 節點
def link_case_to_reference(tx, case_id):
    tx.run(
        "MATCH (c:Case {id: $case_id}), (r:ReferenceNode {name: '參考用判決書'}) "
        "MERGE (r)-[:參考用資料]->(c)",
        case_id=case_id
    )
    print(f"將{case_id}連結到 參考用判決書")

def delete_all_nodes(tx):
    tx.run("MATCH (n) DETACH DELETE n")
    print("Delete All Node")

# 函數：創建 "參考資料" 節點並將 "起訴書相關法條" 和 "參考用判決書" 連接到它
def create_and_link_reference_data_node(tx):
    tx.run("MERGE (ref:ReferenceData {name: '參考資料'})")
    tx.run("MATCH (ln:LawNode {name: '起訴書相關法條'}), (ref:ReferenceData {name: '參考資料'}) "
           "MERGE (ref)-[:中華民國民法法條]->(ln)")
    tx.run("MATCH (rn:ReferenceNode {name: '參考用判決書'}), (ref:ReferenceData {name: '參考資料'}) "
           "MERGE (ref)-[:範例判決書]->(rn)")
    print("創建 參考資料 節點並將 起訴書相關法條 和 參考用判決書 連接到它")

# 加載文檔並解析
with open('statute.txt', 'r', encoding='utf-8') as file:
    content = file.read()

with driver.session() as session:
    session.execute_write(delete_all_nodes)

# 使用 """ 分割法條和口語化解釋
sections = content.split('"""')
for section in sections:
    match = re.search(r"第 (\d+-?\d*) 條\n(.*?)\n口語化解釋:\s*(.*)", section, re.S)
    if match:
        statute_number = match.group(1).strip()
        statute_text = match.group(2).strip()
        explanation_text = match.group(3).strip()

        statute_id = f"民法第{statute_number}條"

        with driver.session() as session:
            session.execute_write(create_statute_and_explanation, statute_id, statute_text, explanation_text)

# 加載範例案件並解析
with open('example_cases.txt', 'r', encoding='utf-8') as file2:
    content2 = file2.read()

cases = [case.strip() for case in content2.split('"') if case.strip()]

# 創建和連接所有節點
with driver.session() as session:
    session.execute_write(create_law_node)
    
    # 連接所有 Statute 節點到 "起訴書相關法條"
    session.execute_write(link_statutes)
    session.execute_write(create_reference_node)

    for i, case in enumerate(cases):
        case_id = f"Case{i+1}"
        match = re.search(r'一、(.*?)二、(.*)', case, re.S)
        if match:
            fact_text = match.group(1).strip()
            remaining_text = match.group(2).strip()
            comp_match = re.search(r'\（\s*一\s*\）', remaining_text)
            if comp_match:
                legal_text = remaining_text[:comp_match.start()].strip()
                compensation_text = remaining_text[comp_match.start():].strip()
            else:
                legal_text = remaining_text
                compensation_text = ""

            session.execute_write(create_case_node, case_id, case)
            session.execute_write(link_case_to_reference, case_id)
            fact_id = f"Fact{i+1}"
            session.execute_write(create_fact_node, fact_id, fact_text)
            session.execute_write(link_fact_to_case, case_id, fact_id)
            
            if legal_text:
                legal_id = f"Legal{i+1}"
                session.execute_write(create_and_link_legal_node, legal_id, legal_text)
                session.execute_write(link_legal_to_case, case_id, legal_id)
            
            # Create and link the "賠償細項" node
            if compensation_text:
                comp_id = f"Compensation{i+1}"
                session.execute_write(create_compensation_node, comp_id, compensation_text)
                session.execute_write(link_compensation_to_case, case_id, comp_id)

                # Update the regular expression to handle both half-width and full-width brackets
                comp_items = re.findall(r'[（(]([^）)]+)[）)]\s*(.*?)(?=[（(]\w+[）)]|$)', compensation_text, re.S)

                for j, (item_label, item_text) in enumerate(comp_items):
                    item_id = f"CompItem{i+1}_{j+1}"
                    session.execute_write(create_comp_item_node, item_id, item_text.strip())
                    session.execute_write(link_comp_item_to_comp, comp_id, item_id)

    # 創建並連接 "參考資料" 節點
    session.execute_write(create_and_link_reference_data_node)
