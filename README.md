Project_heart_atttack.ipynb - файл ноутбука с проектом
submission.csv - предсказания на тестовых данных
app папка - папка с fastapi приложением, предсказывающим риск сердечного приступа по полученным данным

Для запуска приложения:
1. Клонировать репозиторий
2. Установить зависимости
pip install fastapi uvicorn jinja2 python-multipart pandas scikit-learn joblib
3. Убедиться что в папке есть обученная модель model.joblib
4. Запустить приложение через python
python app.py
5. Открыть веб интерфейс http://localhost:8000/
6. Выбрать файл csv (по кнопке) и нажать на кнопку "Предсказать"
