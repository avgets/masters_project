import streamlit as st
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore
from uuid import uuid4
import os
os.environ["PADDLE_PDX_CACHE_HOME"] = "./models"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

@st.cache_resource
def get_processing_semaphore():
    return BoundedSemaphore(2)

PAGE_PREVIEW_WIDTH = 400


st.set_page_config(page_title="PDF → Таблицы", page_icon="📊", layout="centered")


st.title("📊 Извлечение основных таблиц финансовой отчетности из PDF")
st.write(
    "Загрузите отчетность в формате PDF — приложение найдет, распознает нужные таблицы",
    "и предложит скачать Excel-файл."
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if st.session_state.processed_files:
    st.markdown("### Готовые файлы")

    for idx, item in enumerate(st.session_state.processed_files):
        col_pdf, col_excel, col_btn = st.columns([2.2, 2.2, 1.6], gap="small")

        with col_pdf:
            st.write(f"📄 **PDF:** {item['pdf_name']}")

        with col_excel:
            st.write(f"📊 **Excel:** {item['excel_name']}")

        with col_btn:
            if item["excel_bytes"] is not None:
                st.download_button(
                label="📥 Скачать",
                data=item["excel_bytes"],
                file_name=item["excel_name"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{idx}_{item['excel_name']}",
                use_container_width=True,
            )
            else:
                st.write("—")

        st.divider()


uploaded = st.file_uploader(
    "Выберите PDF-файл",
    type=["pdf"],
    key=f"uploader_{st.session_state.uploader_key}",
)


log_col, preview_col = st.columns([1, 1], gap="medium")


with log_col:
    log_box = st.empty()

with preview_col:
    image_box = st.empty()


def make_logger(log_box):
    def log(message: str):
        with log_box.container():
            st.markdown("### Статус обработки")
            st.write(message)

    return log


if uploaded is not None:

    sem = get_processing_semaphore()
    acquired = False

    try:

        acquired = sem.acquire(blocking=False)
        if not acquired:
            st.error("Сервер сейчас обрабатывает 2 PDF одновременно. Попробуйте через минуту.")
            st.stop()


        with log_col:
            with st.spinner("Идёт обработка...", show_time=False):

                log = make_logger(log_box)
                log("⏳ Загрузка зависимостей...")

                from pipeline import process_pdf_file

         
                with TemporaryDirectory() as tmp_dir:

                    tmp_dir = Path(tmp_dir)

                    pdf_uid = uuid4().hex
                    input_pdf = tmp_dir / f"{pdf_uid}.pdf"

                    safe_stem = str(Path(uploaded.name).stem)[:100]
                    output_name = f"{safe_stem}_tables.xlsx"
                    output_xlsx = tmp_dir / output_name

                    pdf_bytes = uploaded.read()
                    input_pdf.write_bytes(pdf_bytes)

                    result = process_pdf_file(input_pdf, output_xlsx, tmp_dir, log)

                    if result is None:
                        log("В загруженном файле основных таблиц финансовой отчетности не найдено!")

                        st.session_state.processed_files.append(
                        {
                        "pdf_name": uploaded.name,
                        "excel_name": "Основных таблиц финансовой отчетности не найдено!",
                        "excel_bytes": None,
                        }
                        )
                    else:
                        excel_bytes = output_xlsx.read_bytes()

                        st.session_state.processed_files.append(
                        {
                        "pdf_name": uploaded.name,
                        "excel_name": output_name,
                        "excel_bytes": excel_bytes,
                        }
                        )
        st.session_state.uploader_key += 1
        st.rerun()

    except Exception as e:
        image_box.empty()
        with log_box.container():
            st.markdown("### Статус обработки")
            st.error(f"Ошибка: {e}")

    finally:
        if acquired:
            sem.release()