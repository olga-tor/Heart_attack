import re
from typing import List

import joblib
import pandas as pd
from fastapi import HTTPException

LEVEL = 0.3

def to_snake_case(column_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1\2', column_name)
    s2 = re.sub(r'[ ]+', '_', s1)
    return s2.lower().strip('_')

def init_model():
    """Инициализация модели с обработкой ошибок"""
    try:
        model = joblib.load("model.joblib")
        if model is None:
            raise ValueError("Модель не найдена в файле")

        print(f"✅ Модель загружена")
        return model

    except FileNotFoundError:
        print("❌ Файл model.joblib не найден")
        raise HTTPException(status_code=500, detail="Модель не найдена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {e}")


class PredictionService:
    """Сервис для работы с предсказаниями"""

    def __init__(self):
        self.model = init_model()

    def _validate_dataframe(self, test_data: pd.DataFrame):
        """Валидация DataFrame"""
        if test_data.empty:
            raise ValueError("CSV файл пустой")

    def predict_csv(self, file) -> List:
        """Предсказание для CSV файла"""
        try:
            # Читаем CSV файл
            test_data = pd.read_csv(file)

            # Валидация данных
            self._validate_dataframe(test_data)
            print(f"📁 Прочитано {len(test_data)} строк, {len(test_data.columns)} столбцов")

            # Выполняем предсказани

            test_data.columns = [to_snake_case(col) for col in test_data.columns]
            probs = self.model.predict_proba(test_data)[:, 1]

            # Применяем порог
            predictions = (probs >= LEVEL).astype(int)
            print(f"🎯 Сделано {len(predictions)} предсказаний")

            return predictions.tolist()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")
