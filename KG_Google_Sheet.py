from KG_Generate import generate_lawsuit
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv
load_dotenv()
# Google Sheets API 配置
SERVICE_ACCOUNT_FILE = os.getenv("PATH_TO_GOOGLE_JSON")
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
# 試算表 ID 和範圍
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID_2")
RANGE_READ = 'Sheet1!A:A'  # 讀取 A 欄
RANGE_WRITE = 'Sheet1!B1'  # 從 B1 開始寫入

# 初始化 Google Sheets 客戶端
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

# 讀取試算表數據並逐條生成結果
def read_and_write_sheets():
    # 讀取 A 欄數據
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_READ).execute()
    values = result.get("values", [])

    if not values:
        print("試算表中沒有可用的數據")
        return

    for i, row in enumerate(values, start=1):  # start=1 表示試算表行數從 1 開始
        user_input = row[0] if row else ""
        if not user_input.strip():
            continue  # 跳過空行

        print(f"正在處理第 {i} 行數據...")
        try:
            lawsuit_draft = generate_lawsuit(user_input)
        except Exception as e:
            print(f"第 {i} 行生成失敗: {e}")
            lawsuit_draft = "生成失敗"

        # 寫入到 B 欄
        write_range = f"Sheet1!B{i}"
        sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=write_range,
            valueInputOption="RAW",
            body={"values": [[lawsuit_draft]]}
        ).execute()
        print(f"第 {i} 行結果已寫入試算表。")

# 執行程序
if __name__ == "__main__":
    read_and_write_sheets()