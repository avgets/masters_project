import streamlit as st
import tempfile
from pathlib import Path
from io import BytesIO
import pandas as pd


st.set_page_config(
    page_title="PDF → Таблицы",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Извлечение таблиц из PDF")
st.write(
    "Загрузите PDF-документ — приложение распознает таблицы с помощью OCR "
    "и предложит скачать Excel-файл. Можно загружать несколько файлов подряд."
)


# ── Загрузка модели (один раз на весь процесс) ───────────────────────────────
@st.cache_resource(
    show_spinner="Загрузка модели OCR… при первом запуске это займёт около минуты"
)
def load_converter():
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
            status_cb(f"📄 Страница {i}: найдено таблиц — **{len(tables)}**")

            for table in tables:
                df = table.export_to_dataframe(doc=doc)
                sheet_name = f"Page{i}_Table{len(all_tables) + 1}"
                all_tables.append((sheet_name, df))

    if not all_tables:
        status_cb("⚠️ Таблицы не обнаружены ни на одной странице.")
        return None

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in all_tables:
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            status_cb(f"📝 Записан лист **{safe_name}**")

    status_cb(f"🎉 Готово! Всего таблиц: **{len(all_tables)}**")
    return output.getvalue()


# ── Инициализация session_state ───────────────────────────────────────────────
# results — список словарей с результатами всех обработанных файлов.
# processed_ids — множество file_id, для быстрой проверки "уже обработан?".
if "results" not in st.session_state:
    st.session_state.results = []          # [{pdf_name, excel_name, excel_bytes}, ...]
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set() # {file_id, ...}


# ── Предзагрузка модели ───────────────────────────────────────────────────────
load_converter()
st.success("✅ Модель загружена и готова к работе.", icon="🤖")


# ── Блок результатов (всегда вверху) ─────────────────────────────────────────
if st.session_state.results:
    st.divider()
    st.subheader("📁 Готовые результаты")
    for item in st.session_state.results:
        col_label, col_btn = st.columns([3, 2])
        with col_label:
            st.markdown(f"📄 **{item['pdf_name']}** → `{item['excel_name']}`")
        with col_btn:
            st.download_button(
                label="📥 Скачать Excel",
                data=item["excel_bytes"],
                file_name=item["excel_name"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                # Уникальный key обязателен, иначе Streamlit ругается на дублирующиеся виджеты
                key=f"dl_{item['file_id']}",
            )


# ── Загрузка нового файла ─────────────────────────────────────────────────────
st.divider()
st.subheader("📤 Загрузить новый файл")

uploaded_file = st.file_uploader(
    "Выберите PDF-файл",
    type=["pdf"],
    help="Поддерживаются многостраничные PDF. Каждая таблица — на отдельном листе Excel.",
)

if uploaded_file is not None:
    file_id = uploaded_file.file_id

    # Обрабатываем только новые файлы — нажатия кнопок скачивания не вызывают обработку
    if file_id not in st.session_state.processed_ids:

        with st.status("⏳ Обрабатываю документ…", expanded=True) as status_container:
            try:
                excel_bytes = convert_pdf_bytes_to_excel(
                    pdf_bytes=uploaded_file.read(),
                    status_cb=lambda msg: st.write(msg),
                )
                if excel_bytes:
                    excel_name = Path(uploaded_file.name).stem + "_tables.xlsx"
                    # Сохраняем в историю
                    st.session_state.results.append({
                        "file_id": file_id,
                        "pdf_name": uploaded_file.name,
                        "excel_name": excel_name,
                        "excel_bytes": excel_bytes,
                    })
                    st.session_state.processed_ids.add(file_id)
                    status_container.update(
                        label="✅ Обработка завершена!",
                        state="complete",
                        expanded=False,
                    )
                    # Перезапускаем страницу: статус исчезает, новый результат
                    # появляется в блоке "Готовые результаты" вверху
                    st.rerun()
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
