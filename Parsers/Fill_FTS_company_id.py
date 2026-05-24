from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, parse_qs
import pandas as pd
import requests
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PATH_COMPANIES = SCRIPT_DIR / 'Companies.xlsx'

## https://bo.nalog.gov.ru/nbo/organizations/1473509/bfo/ <--- JSON с готовой отчетностью

df =  pd.read_excel(PATH_COMPANIES, dtype = {'Наименование': str, 'ИНН': str, 'Код': str, 'Ссылка на карточку': str, 'Последняя активность': str,  'Код ФНС': str})

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
playwright = sync_playwright().start()
browser = playwright.chromium.launch(headless=True)
context = browser.new_context(
    user_agent=user_agent,
    viewport={'width': 1920, 'height': 1080})
page = context.new_page()
page2 = context.new_page()

try:

    for i, row in df.iterrows():

        if not False: #pd.notna(row['Название']) :

            inn = row['ИНН']
            name = row['Наименование']
            print(f"{i} : {inn}: {name}")
            page.goto(f'https://bo.nalog.gov.ru/search?query={inn}',wait_until = 'domcontentloaded')
            page.wait_for_timeout(3000)

            result = page.wait_for_function("""
                () => {
                    if (document.querySelector("a.results-search-table-row")) {return "element";}
                    if (document.body.innerText.includes("По заданным критериям поиска бухгалтерская (финансовая) отчетность не найдена")) {return "text";}
                    return null;
                }
                """, timeout=10000)

            if result.json_value() == "element":
                link = page.locator("a.results-search-table-row")
                href = link.get_attribute('href')
                parts = href.split('/')
                number_part = parts[-1]
                df.loc[i,'Код ФНС'] = number_part
                print(href)

                page2.goto(f'https://bo.nalog.gov.ru/organizations-card/{number_part}', wait_until='load')

                result2 = page2.wait_for_function("""
                () => {
                    if (document.querySelector("div.grid-reports-header-top__period")) {return "element";}
                    return null;
                }
                """, timeout=10000)

                if result2.json_value() == "element":
                    link = page2.locator("div.grid-reports-header-top__period").locator("button.button_primary")
                    year = link.text_content()
                    print(f"Последняя отчетность за {year} год")
                    df.loc[i,'Отчетность ФНС'] = year

            else:
                print("Финансовая отчетность не найдена")

    df.to_excel(PATH_COMPANIES, index=False)

finally:
    browser.close()
    playwright.stop()