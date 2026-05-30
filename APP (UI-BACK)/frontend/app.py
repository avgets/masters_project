import os
import time
from pathlib import Path

import requests
import streamlit as st
from prometheus_client import Counter, Gauge, Histogram, start_http_server

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
APP_METRICS_PORT = int(os.getenv("APP_METRICS_PORT", "9101"))
PAGE_PREVIEW_WIDTH = 400


@st.cache_resource
def start_metrics_server_once() -> bool:
    start_http_server(APP_METRICS_PORT)
    return True


@st.cache_resource
def get_metrics():
    return {
        "PROCESS_REQUESTS_TOTAL": Counter(
            "app_pdf_process_requests_total",
            "Total number of PDF processing requests sent from app to inference",
        ),
        "PROCESS_SUCCESS_TOTAL": Counter(
            "app_pdf_process_success_total",
            "Total number of successful PDF processing requests in app",
        ),
        "PROCESS_ERRORS_TOTAL": Counter(
            "app_pdf_process_errors_total",
            "Total number of failed PDF processing requests in app",
        ),
        "IN_PROGRESS_GAUGE": Gauge(
            "app_pdf_requests_in_progress",
            "Current number of user requests in progress on app side",
        ),
        "REQUEST_DURATION_SECONDS": Histogram(
            "app_backend_request_duration_seconds",
            "End-to-end duration of app -> inference -> app requests",
            buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600, 1200),
        ),
    }


def ensure_session_state() -> None:
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    if "last_uploaded_signature" not in st.session_state:
        st.session_state.last_uploaded_signature = None


def make_output_filename(pdf_name: str) -> str:
    safe_stem = str(Path(pdf_name).stem)[:100]
    return f"{safe_stem}_tables.xlsx"


def render_processed_files() -> None:
    if not st.session_state.processed_files:
        return

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


def make_logger(log_box):
    def log(message: str):
        with log_box.container():
            st.markdown("### Статус обработки")
            st.write(message)

    return log


def extract_error_message(response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
            if detail is not None:
                return str(detail)
    except Exception:
        pass

    text = response.text.strip()
    if text:
        return text

    return f"Ошибка backend: HTTP {response.status_code}"


def call_backend_process(pdf_name: str, pdf_bytes: bytes) -> requests.Response:
    files = {
        "file": (pdf_name, pdf_bytes, "application/pdf"),
    }
    return requests.post(
        f"{BACKEND_URL}/process",
        files=files,
        timeout=None,
    )


def process_uploaded_file(uploaded, log_box, metrics) -> None:
    pdf_name = uploaded.name
    pdf_bytes = uploaded.getvalue()

    if not pdf_bytes:
        with log_box.container():
            st.markdown("### Статус обработки")
            st.error("Ошибка: загруженный файл пустой.")
        return

    log = make_logger(log_box)
    metrics["PROCESS_REQUESTS_TOTAL"].inc()

    with metrics["IN_PROGRESS_GAUGE"].track_inprogress():
        start_ts = time.perf_counter()

        try:
            log("⏳ Отправка PDF в backend...")
            response = call_backend_process(pdf_name, pdf_bytes)
            metrics["REQUEST_DURATION_SECONDS"].observe(time.perf_counter() - start_ts)

            if response.status_code == 200:
                output_name = make_output_filename(pdf_name)
                excel_bytes = response.content

                st.session_state.processed_files.append(
                    {
                        "pdf_name": pdf_name,
                        "excel_name": output_name,
                        "excel_bytes": excel_bytes,
                    }
                )

                metrics["PROCESS_SUCCESS_TOTAL"].inc()
                st.session_state.uploader_key += 1
                st.session_state.last_uploaded_signature = None
                st.rerun()

            elif response.status_code == 429:
                metrics["PROCESS_ERRORS_TOTAL"].inc()
                message = extract_error_message(response)
                log(f"⚠️ {message}")
                return

            elif response.status_code == 400:
                metrics["PROCESS_ERRORS_TOTAL"].inc()
                message = extract_error_message(response)
                with log_box.container():
                    st.markdown("### Статус обработки")
                    st.error(message)
                return

            else:
                metrics["PROCESS_ERRORS_TOTAL"].inc()
                message = extract_error_message(response)
                with log_box.container():
                    st.markdown("### Статус обработки")
                    st.error(message)
                return

        except requests.RequestException as exc:
            metrics["REQUEST_DURATION_SECONDS"].observe(time.perf_counter() - start_ts)
            metrics["PROCESS_ERRORS_TOTAL"].inc()
            with log_box.container():
                st.markdown("### Статус обработки")
                st.error(f"Ошибка связи с backend: {exc}")

        except Exception as exc:
            metrics["REQUEST_DURATION_SECONDS"].observe(time.perf_counter() - start_ts)
            metrics["PROCESS_ERRORS_TOTAL"].inc()
            with log_box.container():
                st.markdown("### Статус обработки")
                st.error(f"Ошибка: {exc}")


def main() -> None:
    st.set_page_config(page_title="PDF → Таблицы", page_icon="📊", layout="centered")

    start_metrics_server_once()
    metrics = get_metrics()
    ensure_session_state()

    st.title("📊 Извлечение основных таблиц финансовой отчетности из PDF")
    st.write("Загрузите отчетность в формате PDF — приложение найдет, распознает нужные таблицы и предложит скачать Excel-файл.")

    render_processed_files()

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
        file_signature = (uploaded.name, uploaded.size)

        if st.session_state.last_uploaded_signature != file_signature:
            st.session_state.last_uploaded_signature = file_signature

            with log_col:
                with st.spinner("Идёт обработка...", show_time=False):
                    process_uploaded_file(uploaded, log_box, metrics)
        else:
            with log_box.container():
                st.markdown("### Статус обработки")
                st.info("Файл загружен и уже находится в обработке или был недавно обработан.")

    else:
        st.session_state.last_uploaded_signature = None
        image_box.empty()


if __name__ == "__main__":
    main()