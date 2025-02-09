from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru").to(device)
model.eval()


def encode(input):
    if isinstance(input, str):
        input = [input]
    encoded_input = tokenizer(input, padding=True, truncation=True, max_length=256, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings.cpu().numpy()

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask



#Sentences we want sentence embeddings for
#sentences = ['Привет! Как твои дела?',
#             'А правда, что 42 твое любимое число?']


#Tokenize sentences
#Perform pooling. In this case, mean pooling
#sentence_embeddings = encode(sentences)
#print("Shape of embeddings:", sentence_embeddings.shape)
#print("Sample embeddings:", sentence_embeddings[0][:10]) # первые 10 значений