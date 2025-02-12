from db_operations import *
from text_preprocess import *
from embedding_model import encode
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Конфигурация
COSINE_THRESHOLD = 0.5  # Порог косинусной схожести
TOP_K_RESULTS = 5  # Количество возвращаемых результатов


def generate_ngrams(words, n=2):
    """Генерация биграмм из списка слов."""
    return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]


def search_with_threshold(query_embedding, index, threshold):
    """Поиск с фильтрацией по порогу схожести."""
    distances, indices = index.search(query_embedding, index.ntotal)
    mask = distances[0] >= threshold
    return indices[0][mask], distances[0][mask]


# Загрузка индекса и метаданных
embeddings, metadata, index = load_index()

# Пример запроса
query = "Сколько калорий содержится в 7up и actimel вишня?"
processed_words = preprocess_text(query)
print("Обработанные слова:", processed_words)

# Генерация биграмм
bigrams = generate_ngrams(processed_words, n=2)
print("Биграммы:", bigrams)

# Объединение слов и биграмм
search_terms = processed_words + bigrams
if not search_terms:
    print("Запрос не содержит значимых терминов.")
    exit()

# Пакетное кодирование всех терминов
try:
    term_embeddings = encode(search_terms)
    faiss.normalize_L2(term_embeddings)
except Exception as e:
    print(f"Ошибка кодирования: {e}")
    exit()

# Поиск по всем эмбеддингам
all_results = []
for i, term in enumerate(search_terms):
    print(f"\nПоиск по термину: '{term}'")
    indices, distances = search_with_threshold(term_embeddings[i:i + 1], index, COSINE_THRESHOLD)

    for idx, dist in zip(indices, distances):
        all_results.append((idx, dist, term))

# Устранение дубликатов и сортировка
unique_results = {}
for idx, dist, term in all_results:
    if idx not in unique_results or dist > unique_results[idx][0]:
        unique_results[idx] = (dist, term)

sorted_results = sorted(unique_results.items(), key=lambda x: x[1][0], reverse=True)

# Вывод результатов
print(f"\nТоп-{TOP_K_RESULTS} релевантных результатов (порог: {COSINE_THRESHOLD}):")
for rank, (idx, (dist, term)) in enumerate(sorted_results[:TOP_K_RESULTS], 1):
    product = metadata[idx]
    print(f"\nРезультат {rank}:")
    print(f"Термин: '{term}'")
    print(f"Название: {product['name']}")
    print(f"Пищевая ценность: {product['nutrition']}")
    print(f"Схожесть: {dist:.4f}")
    print('-' * 50)