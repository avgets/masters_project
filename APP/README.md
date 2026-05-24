# PDF → Таблицы (Streamlit)

Приложение для распознавания таблиц из PDF с помощью Docling + EasyOCR.

## Локальный запуск

### 1. Установить зависимости

```bash
pip install -r requirements.txt
```

**Дополнительно для pdf2image:**
- **Windows**: скачать Poppler https://github.com/oschwartz10612/poppler-windows/releases
  и добавить `bin/` в PATH
- **Ubuntu/Debian**: `sudo apt-get install poppler-utils`

### 2. Запустить приложение

```bash
streamlit run app.py
```

Откроется в браузере по адресу http://localhost:8501

---

## Следующий шаг — Docker

Структура уже готова к контейнеризации:
- модель загружается через `@st.cache_resource`
- PDF принимается как `bytes`, не требует доступа к файловой системе
- Excel передаётся в браузер через `BytesIO`, без сохранения на диск

Dockerfile будет добавлен на следующем шаге.
