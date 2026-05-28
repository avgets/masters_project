# Development of a Machine Learning-Based Intelligent System for Recognition of Financial Statements from Scanned Documents
Repository for HSE masters diploma

Проект включает в себя следующие оснвоные блоки:
1) Парсеры сайтов e-disclosure.ru и bo.nalog.ru
2) Инструмент для human-разметки собранного датасета с возможность генерации автоматической предразметки
3) Ноутбук для evaluation предобученных моделей Table Detection
4) Ноутбук для fine-tuning моделей Table Detection на собранном датасете
5) Реализацию и обучение BiLSTM классификатора типов финансовых таблиц по выходу OCR
   Нормализованная матрица ошибок классификатора в условиях сильного дисбаланса классов ![Нормализованная матриаца ошибок][(https://drive.google.com/file/d/10liyhI30I1-dnRkOlkHWdc4jH7G80AQT/view?usp=drive_link)]
7) Ноутбук с базовым пайплайном обработки pdf-документа
8) Код серверног приложения streamlit
9) Ссылки на собранный датасет:  
   8.1) Полный датасет изображений страниц документов регулярной финансовой отчетности эмитентов (model evaluation): <a href="https://drive.google.com/drive/folders/1HtduuuAqOhfV-yYdlpP_yn3Rgv9ZlUOh" target="_blank">Ссылка</a>  
   8.2) Fine-tuning датасет для YOLA и Microsoft Table Transformer (train/val/test): <a href="https://drive.google.com/drive/folders/10MbXZqhMTkkF3gQXcK_cnZ0wu_c4rjL8" target="_blank">Ссылка</a>
   8.3) Ground Truth (годовая отчетность с сайта ФНС в формате html):  <a href="https://drive.google.com/drive/folders/1CrEx_4Ee_cZoBZPHSqVKlznryJMV1j8O" target="_blank">Ссылка</a>
   
   
