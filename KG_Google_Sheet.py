from KG_Generate import generate_lawsuit
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv
load_dotenv()
# Google Sheets API 配置
SERVICE_ACCOUNT_FILE = os.getenv("PATH_TO_JSON")
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
# 試算表 ID 和範圍
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
RANGE_READ = '工作表1!A:A'  # 讀取 A 欄
RANGE_WRITE = '工作表1!B1'  # 從 B1 開始寫入

# 函數：與 Google Sheets API 交互
def read_and_write_sheets():
    # 認證 Google Sheets API
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    
    # 讀取 A 欄
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_READ).execute()
    values = result.get('values', [])
    
    if not values:
        print("No data found in column A.")
        return

    # 處理每一行輸入
    outputs = []
    for row in values:
        user_input = row[0] if row else ""
        if user_input.strip():
            print(f"Processing: {user_input}")
            lawsuit = generate_lawsuit(user_input)
            outputs.append([lawsuit])
        else:
            outputs.append([""])  # 空行對應空輸出

    # 將生成的起訴書寫入 B 欄
    body = {'values': outputs}
    sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_WRITE,
        valueInputOption="RAW",
        body=body
    ).execute()
    print("Lawsuits written to column B.")

# 執行程序
if __name__ == "__main__":
    read_and_write_sheets()