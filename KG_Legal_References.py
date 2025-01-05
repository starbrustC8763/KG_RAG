from KG_Faiss_Query import get_legal, fetch_statutes_and_explanations  # 從外部模組導入查詢法條及相關資訊的函式
import re  # 用於正則表達式操作
from langchain.chains import LLMChain  # 用於執行 LLM 鏈的核心模組
from langchain.prompts import PromptTemplate  # 用於定義提示模板
from langchain_ollama import OllamaLLM  # 用於調用 Ollama 模型的模組

# 使用者輸入的範例數據
user_data = """
一、事故發生緣由：
被告於民國94年10月10日20時6分許，駕駛車牌號碼3191-XA號自小客車，沿台南縣山上鄉○○村○○○○○道路由東往西方向行駛，於行經明和村明和192之6號前時，原應注意汽車不得逆向行駛，且應注意車前狀況，並減速慢行，作好隨時準備煞車之安全措施，依當時天氣晴朗、路面平坦無缺陷、無障礙物、桅距良好等，並無不能注意之情事，竟仍疏未注意，逆向駛入對向車道，致其上開自小客車車頭與由乙○○所騎乘、後方搭載其妹丙○○，行駛於對向車道之UWL-1855號輕型機車發生對撞，致原告乙○○、丙○○人車倒地。

二、原告受傷情形：
原告乙○○因而受有頭部挫傷合併顏面多處擦傷及牙齒斷裂、下腹部挫傷合併恥骨之兩側下分枝斷裂（骨折）、右側卵巢巧克力囊腫破裂致腹膜炎等傷害，丙○○因而受有顱腦損傷、顱內出血、顏面骨骨折等傷害。

三、請求賠償的事實根據：
（一）丙○○部分：
1. 醫藥費用：共新台幣38,706元。

2. 不能工作之損失：原告因傷至少2個月不能工作，以92年、93年度扣繳憑單給付總額之平均數1個月34,860元計算，共損失69,728元。

3. 精神慰撫金：原告因被告之侵權行為導致顱內出血及顏面骨折，有頭疼、頭暈且記憶力減退之後遺症等現象，均影響生活、工作及女生外貌甚鉅，導致原告精神痛苦不堪，為此爰請求精神賠償300,000元。

（二）乙○○部分:
1. 醫藥費用：共274,874元。

2. 不能工作之損失：原告因傷無法工作期間達6個月以上，以其92年度綜合所得稅各類所得資料可知其每月工作所得為30,157元，以6個月計，則損失約180,942元。

3. 精神慰撫金：原告因被告之侵權行為而導致下腹部挫傷合併恥骨骨折、卵巢破裂合併內出血等傷害，已切除卵巢百分之50且可能導致終生不孕。而生育對多數女性而言乃視為極為重要之天職，若無法生孕，甚至可能造成婚姻之不幸福及家庭之缺憾，因而使原告極其痛苦，爰請求精神慰撫金2,000,000元。

4. 原告乙○○大學畢業，現在豐年豐和企業股份有限公司上班，月薪約30,000元左右，名下無不動產；原告丙○○為二專畢業，受傷之前的月薪約34,000元左右，名下有汽車1輛，無不動產。
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
### 法條及口語化解釋
{statutes_with_explanations}
### 任務
對每一項條文內容與口語化解釋，判斷應引用的法條是否完整解釋案件需求，並輸出法條以及其條文
你只需要照以下的格式輸出，不要自己增加其他內容
### 輸出格式
- 引用法條：
  - 民法第xxx條
    條文
  - 民法第xxx條
    條文
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
    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct-dpo",
                    temperature=0.1,
                    keep_alive=0,
                    num_predict=len(user_data)+200
                    )
    # 創建 LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    legal_reference = llm_chain.run({
        "case_facts": input_data["case_facts"],
        "injury_details": input_data["injury_details"],
        "statutes_with_explanations": statutes_with_explanations_str
    })
    return legal_reference

# 測試：生成法律引用建議
#legal_reference = generate_legal_reference(user_data)
#print(legal_reference)
