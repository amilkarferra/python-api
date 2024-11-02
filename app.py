from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

app = FastAPI()

# Modelo Pydantic para el cuerpo de la solicitud
class CompareRequest(BaseModel):
    description1: str
    description2: str

# Cargar el modelo y el tokenizador preentrenado
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    # Tokenizar el texto y obtener los embeddings
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Obtener el embedding de la primera posición (cls token)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def cosine_similarity_manual(vectorA, vectorB):
    dot_product = np.dot(vectorA, vectorB)
    normA = np.linalg.norm(vectorA)
    normB = np.linalg.norm(vectorB)
    if normA == 0 or normB == 0:
        return 0.0  # Si uno de los vectores es cero, la similitud debe ser cero.
    return dot_product / (normA * normB)

@app.post("/compare", summary="Comparar descripciones", description="Este endpoint compara dos descripciones de imágenes y devuelve un valor de similitud.")
async def compare_descriptions(request: CompareRequest):
    embedding1 = get_embedding(request.description1)
    embedding2 = get_embedding(request.description2)
    # Calcular la similitud utilizando Cosine Similarity manual
    similarity = cosine_similarity_manual(embedding1, embedding2)
    # Convertir el resultado a float de Python
    similarity = float(similarity)
    return {"similarity": similarity}

# Método de Health Check para verificar si la API está funcionando
@app.get("/health", summary="Health Check", description="Verifica si la API está levantada y funcionando correctamente.")
async def health_check():
    return {"status": "OK"}
