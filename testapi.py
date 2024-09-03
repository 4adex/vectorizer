from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional



from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class VectorInputConfig(BaseModel):
    pooling_strategy: str


class VectorInput(BaseModel):
    text: str
    config: Optional[VectorInputConfig] = None


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

def generate_embeddings_single(sentence, config: VectorInputConfig):
    sentences = [sentence]
    sentence_embeddings = generate_embeddings_batch(sentences, config)
    return sentence_embeddings


def generate_embeddings_batch(sentences, config: VectorInputConfig):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.inference_mode():
        model_output = model(**encoded_input)
    
    if config.pooling_strategy == 'mean':
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    else:
        raise ValueError(f"Pooling strategy {config.pooling_strategy} not supported.")
    
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


app = FastAPI()

class SentenceInput(BaseModel):
    sentence: str

@app.post("/vectorize/")
@app.post("/vectorize")
async def vectorize(input: VectorInput):
    try:
        SentenceInput = input.text
        config = input.config
        sentence_embeddings = generate_embeddings_single(SentenceInput, config)
        
        vectorize_handler = sentence_embeddings.cpu().tolist()
        vectorize_handler = vectorize_handler[0]
        
        return {"text":input.text,"vector": vectorize_handler, "dim": len(vectorize_handler)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
