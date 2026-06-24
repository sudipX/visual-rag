import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "/home/kazi/clip-model"

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"CLIP Model is ready. Running on : {device}")


def embed_image(image_path : str) -> np.ndarray:

    # convert a single image into 512 dimensional vector

    #CLIP's ViT expects only 3 channels, so we always convert to RGB.
    image = Image.open(image_path).convert("RGB")
    
    
    inputs = processor(images=image, return_tensors = "pt") #resizes the images into 224*224 and return the pytorch tensor

    inputs = {k:v.to(device) for k,v in inputs.items()} # take the images into same device as the model

    with torch.no_grad():
        output = model.get_image_features(**inputs) #runs only the image encoder part of CLIP, returns a tensor of shape (1,512)

    image_features = output.pooler_output
    embedding = image_features.cpu().numpy()[0] 
    #cpu(): move from GPU to cpu
    #numpy(): convert into numpy
    #[0]: remove the batch dimension, that is (1,512) to (512,)

    embedding = embedding/np.linalg.norm(embedding)

    return embedding #returns (512,)


def embed_text_clip(text : str) ->np.ndarray:
    # convert a short text clip into embeddings
    # Ideally under 60 words as CLIP's text encoder was trained for image captions

    inputs = processor(
        text=[text],
        return_tensors = "pt",
        padding = True,
        truncation = True
    )

    inputs = {k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        output = model.get_text_features(**inputs)

    text_features = output.pooler_output
    embeddings = text_features.cpu().numpy()[0]
    embeddings = embeddings/np.linalg.norm(embeddings)

    return embeddings

# if __name__=="__main__":

#     text_a = "A golder retriever playing fetch in the park"
#     text_b = "A dog catching a ball outside"
#     text_c = "The quaterly earnings report exceeded expectations"

#     vec_a = embed_text_clip(text_a)
#     vec_b = embed_text_clip(text_b)
#     vec_c = embed_text_clip(text_c)

#     print(np.dot(vec_a,vec_b))
#     print(np.dot(vec_a,vec_c))

#     print(vec_a)
#     print(np.shape(vec_a))
#     print(np.linalg.norm(vec_a))






    