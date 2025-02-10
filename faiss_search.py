import os
import faiss
from db_operations import *
from transformers import AutoTokenizer, AutoModel
from embedding_model import device


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Embeddings and metadata load
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    print("Эмбеддинги и метаданные загружены из файлов.")
else:
    products = load_db()
    embeddings, metadata = save_db(products)


#FAISS Index load
INDEX_FILE = 'db_and_weights/faiss_index.index'
d = embeddings.shape[1]

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    print("FAISS индекс загружен из файла.")
else:
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("FAISS индекс создан и сохранён в файл.")


#Index search properties
query = "что можно приготовить из говядины?"
query_embedding = encode([query])
faiss.normalize_L2(query_embedding)
k = 25
distances, indices = index.search(query_embedding, k)









tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/distilrubert-small-cased-conversational")
model = AutoModel.from_pretrained("DeepPavlov/distilrubert-small-cased-conversational").to(device)

system_message = (
    "Ты — помощник ассистент, который умеет и всегда находит продукты питания в предоставленном тебе тексте"
    "Твоя задача — находить и выводить список всех продуктов питания, упомянутых в любом тексте, который тебе предоставят."
    "В ответе необходимо предоставлять продукты, разделённые между собой пробелом"
)

user_message = "Сколько калорий в 7up и actimel вишня?"
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)


input_ids = tokenizer(prompt, return_tensors='pt').to(device)["input_ids"]
outputs = model.generate(input_ids, max_new_tokens=216)
print(f'Prompt:\n{prompt}')
print(f'Continuation:\n{tokenizer.decode(outputs[0])}\n')









print("Результаты поиска:")
for rank, idx in enumerate(indices[0]):
    product = metadata[idx]  # metadata загружены или сохранены ранее
    print(f"\nРезультат {rank + 1}:")
    print(f"Название: {product['name']}")
    print(f"Пищевая ценность: {product['nutrition']}")
    print(f"Схожесть: {distances[0][rank]:.4f}")



