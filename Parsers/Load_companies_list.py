"""
Получает excel-файл со списком ИНН и дополняет его информацией с сайта раскрыия
"""
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, parse_qs
import pandas as pd
import argparse
from pathlib import Path

PATH_COMPANIES_DEFAULT = r"./Companies.xlsx"
PATH_COMPANY_SEARCH = 'https://e-disclosure.ru/poisk-po-kompaniyam'

def valid_excel_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.exists():
        raise argparse.ArgumentTypeError(f"Файл не найден: {path}")

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Это не файл: {path}")

    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise argparse.ArgumentTypeError(
            f"Ожидается Excel-файл (.xlsx или .xls), получено: {path.name}"
        )

    return path

parser = argparse.ArgumentParser(description="Обработка Excel-файла со списком ИНН")
parser.add_argument("excel_file", nargs="?", type=valid_excel_path, default=PATH_COMPANIES_DEFAULT, help=f"Путь к Excel-файлу (по умолчанию {PATH_COMPANIES_DEFAULT})")
args = parser.parse_args()
PATH_COMPANIES = args.excel_file

print(f"Файл валиден: {PATH_COMPANIES }")


df =  pd.read_excel(PATH_COMPANIES, dtype = {'Наименование': str, 'ИНН': str, 'Код': str, 'Ссылка на карточку': str, 'Последняя активность': str})

user_agent = "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
playwright = sync_playwright().start()
browser = playwright.chromium.launch(headless=True)
context = browser.new_context(
    user_agent=user_agent,
    viewport={'width': 1920, 'height': 1080})
page = context.new_page()
#page2 = None
print("Браузер открыт...")

try:
    page.goto(PATH_COMPANY_SEARCH, wait_until = 'domcontentloaded')
    page.wait_for_timeout(3000)


    for i, row in df.iterrows():


        if not False: #pd.notna(row['Название']) :

            inn = row['ИНН']
            print(f"{i+1} : {inn}")
            page.fill('#textfield', inn)
            page.press('#textfield', 'Enter')
            page.locator('#loaderIcon').first.wait_for(state = 'hidden')
            page.wait_for_timeout(250)
            
            SR = page.wait_for_selector('#searchResults')
            first_row = SR.query_selector('table tr:nth-child(2)')
            if first_row:
                link = first_row.query_selector('a')
                #if link:
                href = link.get_attribute('href')
                name = link.inner_text()
                parsed = urlparse(href)
                params = parse_qs(parsed.query)
                last_active_date = first_row.query_selector('td:nth-child(5)').text_content()
                company_id = params.get('id', [None])[0]
                df.loc[i,'Код'] = company_id
                df.loc[i,'Ссылка на карточку'] = r"https://e-disclosure.ru" + href
                df.loc[i,'Наименование'] = name
                df.loc[i,'Последняя активность'] = last_active_date
                print(f"Идентификатор компании на портале раскрытия: {company_id}") 
                
                #url_ifrs = f"https://e-disclosure.ru/portal/files.aspx?id={company_id}&type=4"

                #if page2 is None:
                #    page2 = context.new_page()
                #response_status = None
                #response_url = None
                    
                #page2.goto(url_ifrs, wait_until='commit')
                #final_url = page2.url
                #if final_url == url_ifrs:
                #    print(f"Ссылка на МСФО: {url_ifrs}")
                #else:
                #    print("МСФО не выкладывается")
            

    df.to_excel(PATH_COMPANIES, index=False)

finally:
    browser.close()
    playwright.stop()