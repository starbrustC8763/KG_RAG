from KG_Faiss_Query import get_legal, fetch_statutes_and_explanations  # 從外部模組導入查詢法條及相關資訊的函式
import re  # 用於正則表達式操作
from langchain.chains import LLMChain  # 用於執行 LLM 鏈的核心模組
from langchain.prompts import PromptTemplate  # 用於定義提示模板
from langchain_ollama import OllamaLLM  # 用於調用 Ollama 模型的模組

# 使用者輸入的範例數據
user_data = """
一、事故發生緣由：
被告乙○○於民國96年12月5日18時35分許，騎乘車牌號碼：922-BER號機車，自臺北市○○路由東往西行駛，途經青年新城前時，適原告正在穿越青年路行走，被告乙○○本應注意車前狀況及依規定暫停讓行人先行通過，然而當時天候陰、夜間有照明、柏油路面乾燥無缺陷、無障礙物、視距良好的情況下，被告乙○○竟疏未注意撞及原告，致原告倒地受傷。又事故發生時被告乙○○未滿20歲，被告丙○○為其法定代理人，依法應連帶負賠償之責。

二、原告受傷情形：
原告因本次車禍事故受有頸椎管狹窄併脊髓損傷之傷害，導致下肢乏力、經常跌倒及行動不便，需要持續復健治療，至今仍步態不穩蹣跚，行動能力明顯受到障礙。

三、請求賠償的事實根據：
原告於96年12月5日至97年1月11日在臺北市立聯合醫院住院支出醫療費用新臺幣8,406元，於97年1月14日至98年11月30日至該院接受物理治療自付費用3,966元，合計12,372元，有該院住院醫療費用證明書及門診醫療費用明細表可以證明。
"""

# 定義提示模板，用於生成法律參考判斷的指引
prompt = PromptTemplate(
    input_variables=["case_facts", "injury_details", "compensation_request", "statutes_with_explanations"],
    template="""
你是一位專業的台灣律師，以下是案件的相關資料及可能需要引用的法條資訊，請根據這些資訊提供起訴書所需的法條引用。
### 案件事實
{case_facts}
### 受傷情形
{injury_details}
### 賠償請求
{compensation_request}
### 法條及口語化解釋
{statutes_with_explanations}
### 任務
對每一項條文內容與口語化解釋，判斷應引用的法條是否完整解釋案件需求，判斷不需引用的法條也要解釋為什麼不需引用
### 輸出格式
- 引用法條：
  - 民法第xxx條
    判斷是否要引用:
  - 民法第xxx條
    判斷是否要引用:
- 總結：本案件中，建議引用上述法條進行起訴。
"""
)

# 函數：將使用者輸入拆分為三個部分
def split_input(user_input):
    """
    使用正則表達式將使用者輸入的數據分割為三個部分：
    1. 案件事實
    2. 受傷情形
    3. 賠償請求

    Args:
        user_input (str): 使用者提供的輸入數據。

    Returns:
        dict: 包含三個關鍵部分的字典。
    """
    sections = re.split(r"(一、|二、|三、)", user_input)
    input_dict = {
        "case_facts": sections[2].strip(),
        "injury_details": sections[4].strip(),
        "compensation_request": sections[6].strip()
    }
    return input_dict

# 函數：解析法條列表
def parse_legal_references(legal_references: str) -> list[str]:
    """
    將法條字符串解析為列表。

    Args:
        legal_references (str): 包含多個法條的字符串，每行一個法條。

    Returns:
        list[str]: 法條的列表。
    """
    return legal_references.split("\n")

# 函數：查詢法條及其口語化解釋
def get_statutes_and_explanation(user_data):
    """
    從使用者提供的數據中提取相關的法條及口語化解釋。

    Args:
        user_data (str): 使用者提供的案件數據。

    Returns:
        list[dict]: 每個法條包含其 ID、條文及口語化解釋。
    """
    input_data = split_input(user_data)
    legal_ref = get_legal(input_data["case_facts"], input_data["injury_details"])
    legal_list = parse_legal_references(legal_ref)
    statutes_with_explanations = fetch_statutes_and_explanations(legal_list)
    return statutes_with_explanations

# 函數：格式化法條及其口語化解釋
def format_statutes_and_explanations(statutes_with_explanations):
    """
    將法條和口語化解釋轉換為格式化的字符串。

    Args:
        statutes_with_explanations (list[dict]): 包含法條數據的列表。

    Returns:
        str: 格式化的字符串。
    """
    formatted_output = []
    for statute in statutes_with_explanations:
        formatted_output.append(
            f"法條: {statute['statute_id']}\n"
            f"條文: {statute['statute_text']}\n"
            f"口語化解釋: {statute['explanation_text']}\n"
        )
    return "\n".join(formatted_output)

# 函數：生成法律引用建議
def generate_legal_reference(user_data):
    """
    對使用者輸入的案件數據生成建議引用的法律條文。

    Args:
        user_data (str): 使用者提供的案件數據。

    Returns:
        str: 包含建議法律引用的文本。
    """
    statutes_with_explanations = get_statutes_and_explanation(user_data)
    statutes_with_explanations_str = format_statutes_and_explanations(statutes_with_explanations)
    input_data = split_input(user_data)
    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct-dpo-q6_K",
                    temperature=0.1,
                    keep_alive=0,
                    num_predict=len(user_data)+200
                    )
    # 創建 LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    legal_reference = llm_chain.run({
        "case_facts": input_data["case_facts"],
        "injury_details": input_data["injury_details"],
        "compensation_request": input_data["compensation_request"],
        "statutes_with_explanations": statutes_with_explanations_str
    })
    return legal_reference

# 測試：生成法律引用建議
legal_reference = generate_legal_reference(user_data)
print(legal_reference)
