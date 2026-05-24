import streamlit as st
import tempfile
from pathlib import Path
from io import BytesIO
import pandas as pd

PAGE_PREVIEW_WIDTH = 400


st.set_page_config(page_title="PDF → Таблицы", page_icon="📊", layout="centered")


st.title("📊 Извлечение таблиц из PDF")
st.write(
    "Загрузите PDF-документ — приложение распознает таблицы с помощью OCR "
    "и предложит скачать Excel-файл."
)

for key, default in [
    ("uploader_key", 0),
    ("processed_files", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.processed_files:
    st.markdown("### Готовые файлы")

    for idx, item in enumerate(st.session_state.processed_files):
        col_pdf, col_excel, col_btn = st.columns([2.2, 2.2, 1.6], gap="small")

        with col_pdf:
            st.write(f"📄 **PDF:** {item['pdf_name']}")

        with col_excel:
            st.write(f"📊 **Excel:** {item['excel_name']}")

        with col_btn:
            st.download_button(
                label="📥 Скачать",
                data=item["excel_bytes"],
                file_name=item["excel_name"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{idx}_{item['excel_name']}",
                use_container_width=True,
            )

        st.divider()



@st.cache_resource(show_spinner="Загрузка OCR-модели...")
def load_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(
        lang=["ru"],
        force_full_page_ocr=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


converter = load_converter()


def add_log(log_lines, message):
    log_lines.append(message)


def render_log(log_box, log_lines):
    with log_box.container():
        st.markdown("### Статус обработки")
        for line in log_lines:
            st.write(line)


def process_pdf(pdf_bytes: bytes, log_box, image_box, dpi: int = 200):
    from pdf2image import convert_from_bytes

    log_lines = []
    add_log(log_lines, "⏳ Преобразование PDF в изображения...")
    render_log(log_box, log_lines)

    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    total_pages = len(images)

    if not images:
        image_box.empty()
        return None

    all_tables = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for i, image in enumerate(images, start=1):
            add_log(log_lines, f"🔍 Обрабатываю страницу {i} / {len(images)}…")
            render_log(log_box, log_lines)

            image_box.image(
                image,
                caption=f"Текущая страница: {i} из {total_pages}",
                width=PAGE_PREVIEW_WIDTH,
            )

            img_path = tmp_path / f"page_{i}.png"
            image.save(img_path, "PNG")

            result = converter.convert(img_path)
            doc = result.document
            tables = doc.tables

            add_log(log_lines, f"📄 Страница {i}: найдено таблиц — **{len(tables)}**")
            render_log(log_box, log_lines)

            for table in tables:
                df = table.export_to_dataframe(doc=doc)
                sheet_name = f"Page{i}_Table{len(all_tables) + 1}"
                all_tables.append((sheet_name, df))

    image_box.empty()

    if not all_tables:
        add_log(log_lines, "⚠️ Таблицы не найдены в документе.")
        render_log(log_box, log_lines)
        return None

    add_log(log_lines, "⏳ Сохранение Excel-файла...")
    render_log(log_box, log_lines)

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        for sheet_name, df in all_tables:
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    add_log(log_lines, f"✅ Готово. Извлечено таблиц: {len(all_tables)}")
    render_log(log_box, log_lines)

    return excel_buffer.getvalue(), len(all_tables)

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


if uploaded is not None:
    pdf_bytes = uploaded.read()
    output_name = f"{Path(uploaded.name).stem}_tables.xlsx"

    try:
        with log_col:
            with st.spinner("Идёт обработка...", show_time=False):
                result = process_pdf(pdf_bytes, log_box, image_box)

        if result is None:
            pass
        else:
            excel_bytes, num_tables = result
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