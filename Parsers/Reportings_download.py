import os
import logging
import asyncio
import aiofiles
import zipfile
import rarfile
rarfile.UNRAR_TOOL = r"C:\Users\GaV\Desktop\Рабочий стол\UnRAR.exe"
from stream_unzip import stream_unzip
import pandas as pd
import py7zr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from aiohttp import ClientSession, TCPConnector, ClientError, ClientTimeout
#from stream_unzip import stream_unzip
#import chardet
from pypdf import PdfReader

import sys
import platform
if platform.system() == 'Windows':
    sys.stdout.reconfigure(encoding='utf-8')

import re
from pathlib import Path

ALLOWED_REPORT_TYPES = {
    'Годовая бухгалтерская отчетность (все формы)',
    'Промежуточная бухгалтерская отчетность (все формы)',
    'Годовая консолидированная финансовая отчетность по МСФО или иным международно признанным стандартам'
}

ALLOWED_YEARS = {2021, 2022, 2023, 2024, 2025}

MAX_RETRIES = 3
REQUEST_DELAY = 1.5
REQUEST_TIMEOUT = 30
CONCURRENT_LIMIT = 4

SCRIPT_DIR = Path(__file__).resolve().parent
PATH_REPORTINGS = SCRIPT_DIR / 'output_list_files.xlsx'
FILES_DIR = r"C:\FinRepFiles"

logger = logging.getLogger(__name__)

class ReportDownloader:
    """Скачивание и распаковка отчетов"""
    
    def __init__(self, files_dir: str = FILES_DIR):
        self.files_dir = Path(files_dir)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.session: Optional[ClientSession] = None
        self.semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    async def __aenter__(self):
        """Инициализация HTTP сессии"""
        connector = TCPConnector(limit=CONCURRENT_LIMIT, ssl=True)
        self.session = ClientSession(
            connector=connector,
            timeout=ClientTimeout(total=REQUEST_TIMEOUT),
            headers=self._get_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Закрытие HTTP сессии"""
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Заголовки для запросов"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9',
            'Referer': 'https://e-disclosure.ru/',
        }
    
    
    def _safe_filename(self, filename, max_len = 200):

        

        filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
        dot_index = filename.rfind('.')
    
        if dot_index == -1:
            return filename[:max_len]
    
        base = filename[:dot_index]
        ext = filename[dot_index:]

        return base[:max_len-len(ext)]+ext
    
    async def download_file(self, file_id: str, company_id: str, company_name: str, report_period: str, report_type: str, url: str, file_type: str, file_date: str) -> Tuple[bool, List[str], Optional[int]]:
        """
        Скачать файл и распаковать если архив
        
        Returns:
            (успех, список файлов)
        """
        async with self.semaphore:

            modified_report_dict = {'Промежуточная бухгалтерская отчетность (все формы)':'РСБУ',
                                    'Годовая бухгалтерская отчетность (все формы)':'РСБУ',
                                    'Пояснительная записка (сопутствующая информация) к годовой бухгалтерской отчетности':'РСБУ',
                                    'Сообщение об утверждении годовой бухгалтерской отчетности':'РСБУ',
                                    'Аудиторское заключение':'РСБУ',
                                    'Годовая бухгалтерская отчетность – Баланс (Форма №1)':'РСБУ',
                                    'Годовая бухгалтерская отчетность – Отчет о финансовых результатах (прибылях и убытках) (Форма №2)':'РСБУ',
                                    'Годовая бухгалтерская отчетность – Отчет об изменениях капитала (Форма №3)':'РСБУ',
                                    'Годовая бухгалтерская отчетность – Отчет о движении денежных средств (Форма №4)':'РСБУ',
                                    'Пояснения к годовой бухгалтерской отчетности (Форма №5)':'РСБУ',
                                    'Годовая бухгалтерская отчетность – Отчет о целевом использовании полученных средств (Форма №6)':'РСБУ',
                                    'Годовая сводная бухгалтерская отчетность по РСБУ':'РСБУ сводная',
                                    'Аудиторское заключение  к годовой сводной бухгалтерской отчетности по РСБУ':'СБУ сводная',
                                    'Годовая финансовая отчетность по МСФО (индивидуальная)':'МСФО идивидуальная',
                                    'Промежуточная финансовая отчетность по МСФО (индивидуальная)':'МСФО идивидуальная',
                                    'Годовая консолидированная финансовая отчетность по МСФО или иным международно признанным стандартам':'МСФО консолидированная',
                                    'Промежуточная консолидированная финансовая отчетность по МСФО или иным международно признанным стандартам': 'МСФО консолидированная',
                                    'Аудиторское заключение к годовой консолидированной финансовой отчетности по МСФО или иным международно признанным стандартам':'МСФО консолидированная',
                                    'Дополнительная  информация, касающаяся консолидированной финансовой отчетности':'МСФО консолидированная',
                                    'Иная информация, касающаяся сводной (консолидированной) отчетности':'',
                                    'Учетная политика':''
                                     }
            
            modified_report_type = modified_report_dict[report_type]
            if modified_report_type !='':
                subdir_name = f'[{company_name}]-[{modified_report_type}]-[{report_period}]-[{company_id}]-[{file_id}]'#.replace('""',"''")
                subdir_name = self._safe_filename(subdir_name)
                dir_path = Path(self.files_dir / subdir_name)
                dir_path.mkdir(parents=False, exist_ok=True)
            else:
                dir_path = Path(self.files_dir)
                 
            file_name = f'[{report_type}]-[{file_date}]-[{file_id}].{file_type}'
            file_name = self._safe_filename(file_name)
            filepath = dir_path / file_name

            try:

                if filepath.exists():
                    logger.info(f"Файл {file_id} уже существует")
                else:    
                    logger.info(f"Скачивание файла {file_id}...")
                    success = await self._download(url, filepath)
                    if not success:
                        logger.error(f"Не удалось скачать файл: {url}")
                        return False, [], None
                    else:
                        logger.info(f"Файл {file_id} скачан")
            
            
                if file_type in ['zip', 'rar', '7z']:
                    extracted = await self._extract_archive(filepath, file_id, file_type)
            
                    renamed_extracted = []

                    subdirs = [item for item in extracted if Path(item).is_dir()]
                    files = [item for item in extracted if Path(item).is_file()]
                    print(f"Число файлов в архиве: {len(files)}, число директорий: {len(subdirs)}")


                    for ext_file in files:

                        ext_path = Path(ext_file)
                        new_name = f'[{report_type}]-[{file_date}]-[{file_id}]-{ext_path.name}'
                        new_name = self._safe_filename(new_name, 250 - len(str(dir_path)))
                        new_path = dir_path / new_name

                        if new_path.exists():
                            print(f"Файл уже есть: {str(new_path.resolve())}")
                            #new_path = new_path.with_name(str(new_path.name) + '_' )
                        ext_path.rename(new_path)
                        renamed_extracted.append(str(new_path))
                    #except:
                    #    print(f"Не удалось переименовать файл: {str(ext_path)}")
                    #    renamed_extracted.append(str(ext_path))
                    for dir in subdirs:
                        print(f'Удалаем лишниюю папку {dir}')
                        Path(dir).rmdir()
            
                    logger.info(f"Распаковано файлов: {len(files)}")

                    main_file = renamed_extracted[0] if renamed_extracted else '[Error]'

                    page_count = get_page_count(main_file) if main_file != '[Error]' else None

                    if renamed_extracted:
                        print(f"✅ Успешно скачан [{company_name}]-[{modified_report_type}]-[{report_period}]-[{company_id}]-[{file_id}], страниц: {page_count}", flush=True)
                        print('-'*40)
                        return True, renamed_extracted, page_count
                    else:
                        print(f"❌ Архив пустой или не удалось извлечь [{company_name}]-[{modified_report_type}]-[{report_period}]-[{company_id}]-[{file_id}]", flush=True)
                        print('-'*40)
                        return False, ['[Error]'], None
            
            except Exception as e:
                print(f"❌ Критическая ошибка скачивания: [{company_name}]-[{modified_report_type}]-[{report_period}]-[{company_id}]-[{file_id}]: /n {e}", flush=True)
                print('-'*40)
                return False, ['[Error]'], None
            

            page_count = get_page_count(str(filepath))
            return True, [str(filepath)], page_count
    
    async def _download(self, url: str, filepath: Path) -> bool:
        """Скачать файл с retry"""
        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(REQUEST_DELAY * (attempt + 1))
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(await response.read())
                        return True
                    elif response.status in [429, 503, 502]:
                        logger.warning(f"Статус {response.status}, попытка {attempt + 1}")
                        continue
                    else:
                        logger.error(f"Ошибка HTTP {response.status}")
                        return False
                        
            except (ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Ошибка скачивания (попытка {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    return False
        
        return False
    
    def _file_gen(self, zip_path: str):

        with open(zip_path, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk
    
    async def _extract_archive(self, archive_path: Path, file_id: str, file_type: str) -> List[str]:
        """Распаковать архив (zip, rar, 7z)"""
        extracted = []
        
        try:
            extract_dir = archive_path.parent
            #extract_dir.mkdir(parents=True, exist_ok=True)
            
            if file_type == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:

                    #if len(zf.namelist()) == 0:

                    #    print("------ АЛЬТЕРНАТИВНЫЙ РАСПАКОВШИК -----")

                    #    for file_name, file_size, unzipped_chunks in stream_unzip(self._file_gen(self,archive_path)):
       
                    #        encoding = chardet.detect(file_name)['encoding']
                    #        my_file_name =  self._safe_filename(file_name.decode(encoding))
                    #        print(my_file_name)
                    #        file_path = os.path.join(extract_dir, my_file_name)
            
                    #        with open(file_path, 'wb') as out_file:
                    #            for chunk in unzipped_chunks:
                    #                out_file.write(chunk)

                    #        extracted.append(str(file_path.resolve()))

                    #else:

                        for file_name in zf.namelist():
                            if not file_name.endswith('/'):
                                try:
                                    my_file_name = file_name.encode('cp437').decode('cp866')
                                except:
                                    try:
                                        my_file_name = file_name.encode('latin-1').decode('cp1251')
                                    except:
                                        my_file_name = file_name
                                        
                                my_file_name = self._safe_filename(my_file_name,250-len(str(extract_dir)))
                                new_path = extract_dir / my_file_name
                                old_path = Path(zf.extract(file_name, extract_dir))

                                if old_path.exists() and old_path != new_path:
                                    print("Переименование...")
                                    print(f"Старый путь: {old_path}")
                                    print(f"Новый путь: {new_path}")
                                    old_path.rename(new_path)
                                #else:
                                #    print("--- ПУТИ РАВНЫ ИЛИ ИСХОДНЫЙ ПУТЬ НЕ СУЩЕСТВУЕТ -------")
                                extracted.append(str(new_path.resolve()))
            
            elif file_type == 'rar':
                with rarfile.RarFile(archive_path, 'r') as rf:
                    for name in rf.namelist():
                        rf.extract(name, extract_dir)
                        extracted.append(str(extract_dir / name))
            
            elif file_type == '7z':
                with py7zr.SevenZipFile(archive_path, 'r') as sz:
                    sz.extractall(path=extract_dir)
                    for name in sz.getnames():
                        extracted.append(str(extract_dir / name))
            
            
            archive_path.unlink()
            
            logger.info(f"Архив {file_id} распакован")

            
        except Exception as e:
            logger.error(f"Ошибка распаковки {file_type}  {file_id}: {e}")

        #print(extracted)
        
        return extracted
    
    async def download_batch(self, files: List[Dict], company_id: str) -> Dict:
        """
        Параллельное скачивание нескольких файлов
        Args:
            files: Список файлов из DataFrame (to_dict('records'))
            company_id: ID компании
        """
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        tasks = []
        for f in files:
            task = self.download_file(
                file_id=str(f.get('FileID', '')),
                company_id=company_id,
                url=f.get('Ссылка', ''),
                file_type=f.get('Тип файла', 'zip').lower()
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                stats['failed'] += 1
            else:
                success, _ = result
                if success:
                    stats['downloaded'] += 1
                else:
                    stats['failed'] += 1
        
        logger.info(f"Скачано: {stats['downloaded']}, Ошибок: {stats['failed']}")
        return stats
    
def get_page_count(file_path: str) -> Optional[int]:
    if not file_path or file_path == '[Error]':
        return None

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return None

    if path.suffix.lower() != '.pdf':
        return None

    try:
        reader = PdfReader(str(path))
        return len(reader.pages)
    except Exception as e:
        logger.warning(f"Не удалось определить число страниц для {file_path}: {e}")
        return None


def extract_year(period_value) -> Optional[int]:
    if pd.isna(period_value):
        return None

    match = re.search(r'(20\d{2})', str(period_value))
    if not match:
        return None

    return int(match.group(1))


def should_download(report_type, report_period) -> bool:
    if report_type not in ALLOWED_REPORT_TYPES:
        return False

    year = extract_year(report_period)
    if year not in ALLOWED_YEARS:
        return False

    return True

async def process_batch(jobs, df):
    if not jobs:
        return

    results = await asyncio.gather(
        *[job['task'] for job in jobs],
        return_exceptions=True
    )

    for job, result in zip(jobs, results):
        i = job['index']

        if isinstance(result, Exception):
            print(f"Ошибка в строке {i}: {result}", flush=True)
            df.loc[i, 'Путь к файлу'] = '[Error]'
            continue

        success, files, page_count = result

        if success and files and files[0] != '[Error]':
            df.loc[i, 'Путь к файлу'] = files[0]
            if page_count is not None:
                df.loc[i, 'Число страниц'] = page_count
        else:
            df.loc[i, 'Путь к файлу'] = '[Error]'

    df.to_excel(PATH_REPORTINGS, index=False)
    print(f"💾 Сохранен батч из {len(jobs)} файлов", flush=True)
    
async def main():

    df = pd.read_excel(PATH_REPORTINGS, dtype = {'Тип документа': str, 'Отчетный период': str, 'company_name': str, 'ИНН': str, 'Ссылка на карточку': str, 'Последняя активность': str,'FileID': str})

    path_col = 'Путь к файлу'
    if path_col not in df.columns:
        df[path_col] = pd.NA

    if 'Число страниц' not in df.columns:
        df['Число страниц'] = pd.NA

    batch_size = 30

    filled_mask = (
        df[path_col].notna() &
        df[path_col].astype(str).str.strip().ne('') &
        df[path_col].astype(str).str.strip().ne('[Error]')
    )

    if filled_mask.any():
        start_pos = filled_mask[filled_mask].index[-1] + 1
    else:
        start_pos = 0

    print(f"Обработка с позиции {start_pos}", flush=True)

    async with ReportDownloader(FILES_DIR) as downloader:
        jobs = []

        for i in range(start_pos, len(df)):
            row = df.iloc[i]
            file_id = row['FileID']
            company_id = row['company_id']
            company_name = row['company_name']
            report_period = row['Отчетный период']
            report_type = row['Тип документа']
            url = row['Ссылка']
            file_type = row['Тип файла']
            file_date = row['Дата размещения']

            if not should_download(report_type, report_period):
                continue

            jobs.append({
                'index': i,
                'task': downloader.download_file(
                    file_id=file_id,
                    company_id=company_id,
                    company_name=company_name,
                    report_period=report_period,
                    report_type=report_type,
                    url=url,
                    file_type=file_type,
                    file_date=file_date
                )
            })

            if len(jobs) >= batch_size:
                await process_batch(jobs, df)
                jobs = []

        await process_batch(jobs, df)
 
asyncio.run(main())