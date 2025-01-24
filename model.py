import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Загрузка данных из файлов
places_df = pd.read_excel('/root/mymlproj/DataSet.xlsx')  # Файл с местами
users_df = pd.read_excel('/root/mymlproj/users.xlsx')  # Файл с пользователями

# Объединяем данные
merged_df = users_df.merge(places_df, left_on='visited', right_on='id', how='inner')

# Оставляем только нужные колонки: user_id, data.general.name (как item_id), rating, data.general.tags
merged_df = merged_df[['user_id', 'data.general.name', 'rating', 'data.general.tags']]

# Переименуем колонки для удобства
merged_df.columns = ['user_id', 'item_id', 'rating', 'tags']

# Загружаем данные в Surprise
reader = Reader(rating_scale=(1, 5))  # Укажите шкалу оценок
data = Dataset.load_from_df(merged_df[['user_id', 'item_id', 'rating']], reader)

# Разделяем данные на обучающую и тестовую выборки
trainset, testset = train_test_split(data, test_size=0.25)

# Используем алгоритм SVD (Singular Value Decomposition)
algo = SVD()

# Обучаем модель на обучающей выборке
algo.fit(trainset)

# Предсказываем рейтинги на тестовой выборке
predictions = algo.test(testset)

# Оцениваем качество модели с помощью RMSE
accuracy.rmse(predictions)

# Загрузка предобученной модели и токенизатора RuBERT
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

# Функция для получения эмбеддингов текста с помощью BERT
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Используем эмбеддинги из последнего слоя модели
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Функция для рекомендаций на основе Surprise
def get_surprise_recommendations(user_id, top_n=5):
    all_places = places_df['data.general.name'].unique()  # Все возможные места
    recommendations = []
    for place in all_places:
        pred = algo.predict(user_id, place)
        recommendations.append((place, pred.est))

    # Сортируем рекомендации по предсказанному рейтингу
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# Функция для рекомендаций на основе BERT
def get_bert_recommendations(query, top_n=5):
    # Создаем текстовые данные для поиска (название + теги)
    places_df['text'] = places_df['data.general.name'] + ' ' + places_df['data.general.tags']

    # Получаем эмбеддинги для всех мест
    place_embeddings = get_bert_embeddings(places_df['text'].tolist())

    # Получаем эмбеддинг для запроса
    query_embedding = get_bert_embeddings([query])

    # Вычисляем сходство между запросом и местами
    similarities = cosine_similarity(query_embedding, place_embeddings).flatten()

    # Добавляем сходство в DataFrame
    places_df['similarity'] = similarities

    # Сортируем места по сходству
    recommendations = places_df.sort_values(by='similarity', ascending=False).head(top_n)
    return list(zip(recommendations['data.general.name'], recommendations['similarity']))

# Функция для выбора модели
def get_recommendations(user_id, query=None, top_n=5):
    if query:
        # Если есть запрос, используем BERT
        print("Используем BERT...")
        return get_bert_recommendations(query, top_n)
    else:
        # Если запроса нет, используем Surprise
        print("Используем Surprise...")
        return get_surprise_recommendations(user_id, top_n)


# Пример использования
#user_id = 100  # Замените на ID пользователя

# Пример 1: Рекомендации на основе Surprise (без запроса)
#print("Рекомендации на основе Surprise:")
#recommendations = get_recommendations(user_id)
#for place, score in recommendations:
#    print(f"Место: {place}, Предсказанный рейтинг: {score}")

# Пример 2: Рекомендации на основе BERT (с запросом)
#user_query = "посоветуй парки"
#print(f"\nРекомендации для запроса '{user_query}':")
#recommendations = get_recommendations(user_id, query=user_query)
#for place, similarity in recommendations:
#    print(f"Место: {place}, Сходство: {similarity}")