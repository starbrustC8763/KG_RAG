from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from KG_Faiss_Query import query_faiss, get_statutes_for_case

# 函數：生成引用的法條
def generate_legal_references(case_facts, injury_details):
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

# 定義提示模板
prompt_template = PromptTemplate(
    input_variables=["case_facts", "injury_details", "compensation_request", "legal_references"],
    template="""
你是一名台灣原告律師，需要為代理人提供撰寫起訴狀的必要資訊。請根據以下提供的判決書擷取內容，模擬你作為原告律師會提供給代理人的資訊。使用台灣繁體中文回答，以一般書面書寫口吻呈現，可以日常輕鬆一點。
回答格式:
直接以三個大段落呈現，無需其他開場白:
一、事故發生緣由和原告受傷情形
二、針對這起案件的引用法條
三、請求賠償的事實根據

注意事項:
1. 內容完整性：
   - 保持原判決書中的重要細節，包括法條引用、當事人互動描述、傷勢細節、事故經過、賠償依據等。
   - 不要省略或簡化重要資訊，保持原有的詳細程度。

2. 數據準確性：
   - 保留所有相關數字，如住院天數、無法工作時間等，不做更改或四捨五入。
   - 保留任何金額計算公式，使用阿拉伯數字表示，不轉換為中文數字。

3. 個人資訊處理：
   - 保留地址、姓名、車牌等具體資訊，不要匿名化或更改。
   - 如有提及特定人名或地點，需完整保留。

4. 法律引用：
   - 如有引用其他判決，請保留完整的判決年度和號碼。
   - 保留所有法條引用，即使看起來不口語化。

5. 敘述視角：
   - 使用「原告」或「被告」作為主詞，避免使用「我」或其他第一人稱。
   - 保持客觀的敘述口吻，不加入個人情感或評論。

6. 金額處理：
   - 不要自行計算或加總賠償金額，保持原有的分項列舉。
   - 如原文有總計或相加的金額計算公式，需要原封不動地保留。

7. 立場維持：
   - 避免採用法官的觀點或判決結果，始終從原告律師的角度撰寫，因此不要把判決書當中法官的判決內容寫進去。
   - 即使判決書中某些損害未獲賠償，作為原告律師仍應提出所有可能的求償項目。
   - 因為你是原告的律師，所以一開始要書寫起訴狀的時候，一定都會盡量求償，所以即使判決書中有因主要造成財產損失，不涉及人身傷害，最終法官不予以賠償，但是你身為原告的律師還是要把該求償的書寫書來。

8. 時間順序：
   - 按照事件發生的時間順序來敘述，保持邏輯清晰。

9. 格式一致性：
    - 保持三個大段落的結構，每個段落內容要前後呼應。
    - 使用一致的書寫風格，避免忽然改變敘述方式。

請根據隨後提供的判決書擷取內容，按照上述指示撰寫回應。記住，你是在模擬原告律師提供給代理人的資訊，而不是直接撰寫起訴狀。
### 案件事實：
{case_facts}
### 受傷情形：
{injury_details}
### 引用法條：
{legal_references}
### 賠償請求：
{compensation_request}
"""
)

# 測試輸入數據
input_data = {
    "case_facts": """一、事故發生緣由:
被告於民國105年4月12日13時27分許，駕駛租賃小客車沿新北市某區某路往富國路方向行駛。行經福營路342號前時，被告跨越分向限制線欲繞越前方由原告所駕駛併排於路邊臨時停車後適欲起駛之車輛。被告為閃避對向來車，因而駕車自後追撞原告駕駛車輛左後車尾。當時天候晴朗、日間自然光線、柏油道路乾燥無缺陷或障礙物、視距良好，被告理應注意車前狀況及兩車並行之間隔，隨時採取必要之安全措施，但卻疏未注意而發生事故。""",
    "injury_details": """二、原告受傷情形:
原告因此車禍受有左膝挫傷、半月軟骨受傷等傷害。原告於105年5月2日、7日、7月16日、8月13日、8月29日至醫院門診就診，105年8月2日進行核磁共振造影檢查。根據醫院開立的診斷證明書，原告需休養1個月。""",
    "compensation_request": """三、請求賠償的事實根據:
1. 醫療復健費用190元
2. 車輛修復費用181,144元
3. 交通費用4,500元
4. 休養期間工作收入損失33,000元
5. 慰撫金99,000元
"""
}

# 生成引用的法條
legal_references = generate_legal_references(input_data["case_facts"], input_data["injury_details"])
input_data["legal_references"] = legal_references

# 初始化模型
llm = OllamaLLM(model="jcai/llama3-taide-lx-8b-chat-alpha1:f16")

# 創建 LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# 傳入數據生成起訴書
lawsuit_draft = llm_chain.run({
    "case_facts": input_data["case_facts"],
    "injury_details": input_data["injury_details"],
    "legal_references": legal_references,
    "compensation_request": input_data["compensation_request"]
})
print(lawsuit_draft)
