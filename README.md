# Development of a Machine Learning-Based Intelligent System for Recognition of Financial Statements from Scanned Documents
Repository for HSE masters diploma

Проект включает в себя следующие оснвоные блоки:
1) Парсеры сайтов e-disclosure.ru и bo.nalog.ru
2) Инструмент для human-разметки собранного датасета с возможность генерации автоматической предразметки
3) Ноутбук для evaluation предобученных моделей Table Detection
4) Ноутбук для fine-tuning моделей Table Detection на собранном датасете
5) Реализацию и обучение BiLSTM классификатора типов финансовых таблиц по выходу OCR
6) Ноутбук с базовым пайплайном обработки pdf-документа
7) Код серверног приложения streamlit
8) Ссылки на собранный датасет:  
   8.1) Полный датасет изображений страниц документов регулярной финансовой отчетности эмитентов (model evaluation): <a href="https://drive.google.com/drive/folders/1HtduuuAqOhfV-yYdlpP_yn3Rgv9ZlUOh" target="_blank">Ссылка</a>  
   8.2) Fine-tuning датасет для YOLA (train/val/test): <a href="https://drive.google.com/drive/folders/10MbXZqhMTkkF3gQXcK_cnZ0wu_c4rjL8" target="_blank">Ссылка</a>
   
   
