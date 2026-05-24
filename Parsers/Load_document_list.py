import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from io import StringIO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PATH_REPORTINGS = SCRIPT_DIR / 'output_list_files.xlsx'
PATH_COMPANIES = SCRIPT_DIR / 'Companies.xlsx'

async def load_page(url: str, context, semaphore):
    """Асинхронная загрузка с ограничением параллелизма"""
    async with semaphore:  # Ограничиваем кол-во одновременных задач

        page = await context.new_page()
        
        try:
            print(f"📄 Начало загрузки: {url}")

            start_time = asyncio.get_event_loop().time()
            response_status = None
            response_url = None

            await page.goto(url, wait_until='load')

            final_url = page.url
            if final_url == url:
                print(f"URL открыт: {url}")
            else:
                print(f"URL не обнаружен: {url}")
                elapsed = asyncio.get_event_loop().time() - start_time
                print(f"✅ [{url}] загружен за {elapsed:.2f}s")
                return {
                    'url': url,
                    'title': '',
                    'time': elapsed,
                    'status': 'not found',
                    'data': None
                }


            title = await page.title()
            name = await page.text_content('.infoblock')

            table_element = await page.query_selector("table.files-table")
            if table_element is not None:

                data = await page.evaluate("""
                () => {
                    const table = document.querySelector('table.files-table');
                    const rows = table.querySelectorAll('tr');
                    const result = [];
                    
                    for (let i = 1; i < rows.length; i++) {
                        const row = rows[i];
                        const cells = row.querySelectorAll('td');
                        
                        if (cells.length >= 6) {
                            const fileLink = cells[5].querySelector('a.file-link');
                                           
                            let fileType = '';
                            let fileSize = '';
                            if (fileLink) {
                                const fileText = fileLink.textContent.trim();
                                const parts = fileText.split(',').map(s => s.trim());
                                if (parts.length >= 2) {
                                    fileType = parts[0];
                                    fileSize = parts[1];
                                }
                            }
                            
                            result.push({
                                //'№': cells[0].textContent.trim(),
                                'Тип документа': cells[1].textContent.trim(),
                                'Отчетный период': cells[2].textContent.trim(),
                                'Дата основания': cells[3].textContent.trim(),
                                'Дата размещения': cells[4].textContent.trim(),
                                'Файл': fileLink ? fileLink.textContent.trim() : '',
                                'Тип файла': fileType,             
                                'Размер файла': fileSize,          
                                'Ссылка': fileLink ? fileLink.href : '',
                                'FileID': fileLink ? fileLink.getAttribute('data-fileid') : ''
                            });
                        }
                    }
                    
                    return result;
                    }
                """)

                df = pd.DataFrame(data)                              
                print(f"Найдено {len(df)} файлов")

            
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"✅ [{url}] загружен за {elapsed:.2f}s")
            
            
            return {
                'url': url,
                'title': title,
                'time': elapsed,
                'status': 'success',
                'data': df
            }
        except Exception as e:
            print(f"❌ [{url}] Ошибка: {e}")
            return {'url': url, 'error': str(e), 'status': 'failed', 'data': None}
        finally:
            await page.close()

async def main():

    df_companies =  pd.read_excel(PATH_COMPANIES, dtype = {'Наименование': str, 'ИНН': str, 'Код': str, 'Ссылка на карточку': str, 'Последняя активность': str})
    urls_rsbu = []
    urls_ifrs = []
    
    for i, row in df_companies.iterrows():
        company_id = row['Код']
        company_name = row['Наименование']
        company_inn = row['ИНН']
        company_inn = '' if pd.isna(company_inn) else str(company_inn)
        url_rsbu = f"https://e-disclosure.ru/portal/files.aspx?id={company_id}&type=3"
        url_ifrs = f"https://e-disclosure.ru/portal/files.aspx?id={company_id}&type=4"
        urls_rsbu.append({'id': company_id, 'name': company_name, 'ИНН': company_inn,'url': url_rsbu})
        urls_ifrs.append({'id': company_id, 'name': company_name, 'ИНН': company_inn,'url': url_ifrs})
        if i > 10:
            pass
    
    async with async_playwright() as p:
        start_time = asyncio.get_event_loop().time()
        browser = await p.chromium.launch(headless=True)
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        context = await browser.new_context(    
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080}
        )
        
        semaphore = asyncio.Semaphore(3)
        
        tasks_rsbu = [load_page(item['url'], context, semaphore) for item in urls_rsbu]
        results_rsbu = await asyncio.gather(*tasks_rsbu)

        tasks_ifrs = [load_page(item['url'], context, semaphore) for item in urls_ifrs]
        results_ifrs = await asyncio.gather(*tasks_ifrs)
        
        await browser.close()


    dfs_rsbu = []
    list_urls_rsbu = []
    for item_url, item_result in zip(urls_rsbu, results_rsbu):
        df = item_result['data']
        if df is not None:
            company_id = item_url['id']
            company_name = item_url['name']
            company_inn = item_url['ИНН']
            df['company_id'] = company_id
            df['company_name'] = company_name
            df['ИНН'] = company_inn
            dfs_rsbu.append(df)
        if item_result['status']!='not found':
            list_urls_rsbu.append(item_url['url'])
        else:
            list_urls_rsbu.append('')

    dfs_ifrs = []
    list_urls_ifrs = []
    for item_url, item_result in zip(urls_ifrs, results_ifrs):
        df = item_result['data']
        if df is not None:
            company_id = item_url['id']
            company_name = item_url['name']
            company_inn = str(item_url['ИНН'])
            df['company_id'] = company_id
            df['company_name'] = company_name
            df['ИНН'] = company_inn
            df['ИНН'] = df['ИНН'].astype(str)
            dfs_ifrs.append(df)
        if item_result['status']!='not found':
            list_urls_ifrs.append(item_url['url'])
        else:
            list_urls_ifrs.append('')

    df_companies['Ссылка на отчетность РСБУ'] = list_urls_rsbu
    df_companies['Ссылка на отчетность МСФО'] = list_urls_ifrs
    df_companies.to_excel(PATH_COMPANIES, index=False)

    df_total = pd.concat(dfs_rsbu + dfs_ifrs)
    total_time = asyncio.get_event_loop().time() - start_time
    print(f"\n📊 Общее время загрузки: {total_time:.2f}s")

    df_total.to_excel(PATH_REPORTINGS, index=False)

asyncio.run(main())