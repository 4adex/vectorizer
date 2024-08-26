from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time
# from torch.cuda.amp import autocast


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Demo sentences
sentences = [
    "The cat slept peacefully on the windowsill.",
    "She opened the door to a surprise party in her honor.",
    "The sky turned orange as the sun set behind the mountains.",
    "A mysterious letter arrived in the mail, with no return address.",
    "He brewed a strong cup of coffee to start his day.",
    "The sound of waves crashing on the shore was calming.",
    "She found an old photograph album in the attic.",
    "A bird perched on the fence, singing a cheerful tune.",
    "The old clock in the hallway struck midnight.",
    "He lost his keys again, just as he was about to leave.",
    "The garden was full of blooming flowers in every color.",
    "A soft breeze rustled the leaves on the trees.",
    "They decided to go for a walk in the park after dinner.",
    "The lights flickered during the thunderstorm.",
    "She wrote a heartfelt letter to her best friend.",
    "The children laughed as they played in the rain.",
    "A gentle snow began to fall, covering the ground in white.",
    "He couldn’t stop smiling after hearing the good news.",
    "The aroma of freshly baked bread filled the kitchen.",
    "She spotted a shooting star while gazing at the night sky.",
    "The bus arrived just as he reached the stop.",
    "A rainbow appeared after the heavy rain.",
    "The dog barked excitedly when its owner came home.",
    "She read a book by the fireplace on a chilly evening.",
    "The city lights twinkled in the distance.",
    "He found a seashell on the beach during his morning jog.",
    "The cake she baked turned out perfectly golden.",
    "They watched a movie under the stars in the backyard.",
    "The sound of laughter echoed through the house.",
    "A butterfly landed gently on her shoulder.",
    "He carefully wrapped the gift with a bright ribbon.",
    "The old bookstore had a musty, comforting smell.",
    "They danced together in the living room to their favorite song.",
    "The scent of fresh pine filled the air during their hike.",
    "She wore a warm scarf to keep the winter chill away.",
    "The cat purred contentedly as it curled up on the couch.",
    "He painted the walls a vibrant shade of blue.",
    "The ice cream truck jingled its tune down the street.",
    "She planted a small herb garden on her windowsill.",
    "The kids built a fort out of blankets and pillows.",
    "He wrote his thoughts down in a leather-bound journal.",
    "The rain tapped lightly against the windowpane.",
    "She practiced her guitar late into the night.",
    "The smell of popcorn filled the theater.",
    "He admired the sunrise from the top of the hill.",
    "The library was quiet, except for the turning of pages.",
    "She dipped her toes into the cool, clear water.",
    "The airplane soared above the clouds.",
    "He arranged the flowers in a vase on the table.",
    "The stars twinkled brightly in the night sky."
]



# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', torch_dtype=torch.float16, clean_up_tokenization_spaces = True)
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2', torch_dtype=torch.float16)

# Set the device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# this method is supposed to make the inference faster but it doesn't
# model = torch.compile(model)


#We are noting the time taken in tokenization and inference
start_time = time.time()


# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

# Compute token embeddings, do pooling n normalize
with torch.inference_mode():
    model_output = model(**encoded_input)
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

end_time = time.time()

# Print the results and execution time
print("Sentence embeddings:")
print(sentence_embeddings)
print(f"Execution time: {end_time - start_time:.4f} seconds")
