from fastapi import FastAPI, Query
from model import get_recommendations
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Настройка CORS для разрешения запросов с фронта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем запросы со всех доменов
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы
    allow_headers=["*"],  # Разрешаем все заголовки
)

@app.get("/recommend")
async def recommend(user_id: int, query: str = Query(None)):
    # Получаем рекомендации из модели
    recommendations = get_recommendations(user_id, query)
    
    # Возвращаем рекомендации в формате JSON
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="83.222.27.166", port=5000)