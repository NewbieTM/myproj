#import torch
#print("Доступна CUDA:", torch.cuda.is_available())  # Должно быть True
#print("Версия CUDA:", torch.version.cuda)  # Проверьте версию
#print("Название GPU:", torch.cuda.get_device_name(0))  # Должно показать вашу GPU


# import spacy
# nlp = spacy.load("ru_core_news_lg")
# doc = nlp('Сколько калорий в говядине и морковке?')
# print(doc.ents)

