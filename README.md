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
   8.1) Полный датасет изображений страниц документов регулярной финансовой отчетности эмитентов (model evaluation): https://drive.google.com/drive/folders/1mK3Zf0b8x4OZ2234EWJXl5wE26803QbW
   8.2) Fine-tuning датасет для YOLA (train/val/test): https://drive.google.com/drive/folders/1T5ZliDYQ4chzjjdvhWAU15EFVrsj5xGQ
