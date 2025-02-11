from db_operations import *
from text_preprocess import *
from embedding_model import encode

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


embeddings, metadata, index = load_index()

query = "Сколько калорий содержится в 7up и actimel вишня?"

words = preprocess_text(query)
print(words)
for word in words:
    print(f"Ищем по слову: {word}")
    word_embedding = encode([word])
    faiss.normalize_L2(word_embedding)

    k = 3
    distances, indices = index.search(word_embedding, k)

    for rank, idx in enumerate(indices[0]):
        product = metadata[idx]
        print(f"\nРезультат {rank + 1}:")
        print(f"Название: {product['name']}")
        print(f"Пищевая ценность: {product['nutrition']}")
        print(f"Схожесть: {distances[0][rank]:.4f}")
        print('-' * 50)












