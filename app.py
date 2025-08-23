from typing import Dict, Any

from fastapi import FastAPI, UploadFile, HTTPException

from prediction_service import PredictionService

# Создаем приложение
app = FastAPI(title="ML Prediction API", version="1.0")

# Инициализация сервиса
try:
    prediction_service = PredictionService()
except Exception as e:
    print(f"Не удалось инициализировать сервис: {e}")
    prediction_service = None


@app.get("/")
def root():
    return {"message": "ML Prediction API работает! Используйте /predict для предсказаний"}


@app.get("/health")
def health_check() -> Dict[str, Any]:
    status = "OK" if prediction_service and prediction_service.model else "ERROR"
    return {
        "status": status,
        "service_ready": prediction_service is not None
    }


@app.post("/predict")
def predict_from_csv(file: UploadFile):
    """
    Загрузите CSV файл для предсказания.
    Файл должен содержать числовые признаки без заголовков.
    """
    # Проверка типа файла
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Поддерживаются только CSV файлы")

    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Сервис предсказаний недоступен")

    try:
        predictions = prediction_service.predict_csv(file.file)

        return {
            "filename": file.filename,
            "predictions": predictions,
            "total_predictions": len(predictions),
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as er:
        print(f"Ошибка в endpoint: {er}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
