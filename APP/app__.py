import streamlit as st
import tempfile
from pathlib import Path
from io import BytesIO
import pandas as pd


# ── Страница ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF → Таблицы",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Извлечение таблиц из PDF")
st.write(
    "Загрузите PDF-документ — приложение распознает таблицы с помощью OCR "
    "и предложит скачать Excel-файл."
)


# ── Загрузка модели (один раз на весь процесс) ───────────────────────────────
@st.cache_resource(show_spinner="Загрузка модели OCR… это займёт около минуты при первом запуске")
def load_converter():
    """
    Инициализируем DocumentConverter один раз и кешируем в памяти процесса.
    При следующих запросах к странице Streamlit возвращает тот же объект
    без повторной загрузки.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["ru"], force_full_page_ocr=True)

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


def convert_pdf_bytes_to_excel(pdf_bytes: bytes, status_cb, dpi: int = 200) -> bytes | None:
    """
    Принимает PDF как bytes (из st.file_uploader), возвращает Excel как bytes.
    Не создаёт временных файлов за пределами функции.
    status_cb(msg) — колбэк для обновления UI-статуса во время работы.
    """
    from pdf2image import convert_from_bytes

    converter = load_converter()

    status_cb("🔄 Конвертирую PDF в изображения...")
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    status_cb(f"✅ Получено страниц: **{len(images)}**")

    if not images:
        status_cb("⚠️ Не удалось извлечь страницы из PDF.")
        return None

    all_tables = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for i, image in enumerate(images, start=1):
            status_cb(f"🔍 Обрабатываю страницу {i} / {len(images)}…")
            img_path = tmp_path / f"page_{i}.png"
            image.save(img_path, "PNG")

            result = converter.convert(img_path)
            doc = result.document
            tables = doc.tables

            status_cb(
                f"📄 Страница {i}: найдено таблиц — **{len(tables)}**"
            )

            for table in tables:
                df = table.export_to_dataframe(doc=doc)
                sheet_name = f"Page{i}_Table{len(all_tables) + 1}"
                all_tables.append((sheet_name, df))

    if not all_tables:
        status_cb("⚠️ Таблицы не обнаружены ни на одной странице документа.")
        return None

    # Пишем в буфер памяти, чтобы не трогать файловую систему сервера
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in all_tables:
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            status_cb(f"📝 Записан лист **{safe_name}**")

    status_cb(f"🎉 Готово! Всего таблиц: **{len(all_tables)}**")
    return output.getvalue()


# ── Предзагрузка модели при открытии страницы ────────────────────────────────
load_converter()
st.success("✅ Модель загружена и готова к работе.", icon="🤖")

st.divider()

# ── Загрузка файла ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Выберите PDF-файл",
    type=["pdf"],
    help="Поддерживаются многостраничные PDF. Каждая таблица будет на отдельном листе Excel.",
)

if uploaded_file is not None:
    st.divider()

    excel_bytes = None

    with st.status("⏳ Обрабатываю документ…", expanded=True) as status_container:
        def update_status(msg: str):
            st.write(msg)

        try:
            excel_bytes = convert_pdf_bytes_to_excel(
                pdf_bytes=uploaded_file.read(),
                status_cb=update_status,
            )
            if excel_bytes:
                status_container.update(
                    label="✅ Обработка завершена!",
                    state="complete",
                    expanded=False,
                )
            else:
                status_container.update(
                    label="⚠️ Таблицы не найдены",
                    state="error",
                    expanded=True,
                )
        except Exception as e:
            status_container.update(
                label=f"❌ Ошибка: {e}",
                state="error",
                expanded=True,
            )
            st.exception(e)

    # ── Кнопка скачивания ─────────────────────────────────────────────────────
    if excel_bytes:
        st.balloons()
        excel_filename = Path(uploaded_file.name).stem + "_tables.xlsx"
        st.download_button(
            label="📥 Скачать Excel-файл",
            data=excel_bytes,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
