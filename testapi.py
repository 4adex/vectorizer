from fastapi import FastAPI, HTTPException
from pydantic import BaseModel



from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', torch_dtype=torch.float16, clean_up_tokenization_spaces = True)
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2', torch_dtype=torch.float16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

print("model ready for inference")

def generate_embeddings_single(sentence):
    sentences = [sentence]
    sentence_embeddings = generate_embeddings_batch(sentences)
    return sentence_embeddings


def generate_embeddings_batch(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.inference_mode():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


app = FastAPI()


class SentenceInput(BaseModel):
    sentence: str

@app.post("/vectorize/")
async def vectorize_sentence(input_data: SentenceInput):
    try:
        # Perform inference to get the sentence embedding
        sentence = input_data.sentence
        sentence_embeddings = generate_embeddings_single(sentence)
        
        # # Convert the tensor to a list
        vector_list = sentence_embeddings.cpu().tolist()
        vector_list = vector_list[0]

        return {"vector": vector_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
