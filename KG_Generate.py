from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from KG_Legal_References import generate_legal_reference
import re

# 定義提示模板
prompt_template = PromptTemplate(
    input_variables=["case_facts", "injury_details", "compensation_request", "legal_references"],
    template="""
你是一個台灣原告律師，你要撰寫一份車禍起訴狀，但你只需要根據下列格式進行輸出，並確保每個段落內容完整：
（一）事實概述：完整描述事故經過，事件結果及要求賠償盡量越詳細越好
（二）法律依據：先對每一條我給你的引用法條做判斷，如果確定在這起案件中要引用，列出所有相關法律條文，並對每一條文做出詳細解釋與應用。
  模板：
  - 民法第xxx條第x項：「...法律條文...」。
    - 案件中的應用：本條適用於 [事實情節]，因為 [具體行為] 屬於 [法條描述的範疇]，因此 [解釋為何負責賠償]。
（三）損害項目：列出所有損害項目的金額，並說明對應事實。
  模板：
    損害項目名稱： [損害項目描述]
    金額： [金額數字] 元
    事實根據： [描述此損害項目的原因和依據]
（四）總賠償金額：需要將每一項目的金額列出來並總結所有損害項目，計算總額，並簡述賠償請求的依據。
  模板:
    損害項目總覽：
    總賠償金額： [總金額] 元
    賠償依據：
    依據 [法律條文] 規定，本案中 [被告行為] 對原告造成 [描述損害]，被告應負賠償責任。總賠償金額為 [總金額] 元。
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
user_input="""
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

def split_input(user_input):
    sections = re.split(r"(一、|二、|三、)", user_input)
    input_dict = {
        "case_facts": sections[2].strip(),
        "injury_details": sections[4].strip(),
        "compensation_request": sections[6].strip()
    }
    return input_dict

def generate_lawsuit(user_input):
    input_data=split_input(user_input)
    legal_references = generate_legal_reference(user_input)
    input_data["legal_references"] = legal_references
    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:70b-instruct-dpo-q3_K_S",
                    temperature=0.1,
                    keep_alive=0,
                    num_predict=5000
                    )
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
    return lawsuit_draft
#generate_lawsuit(user_input)