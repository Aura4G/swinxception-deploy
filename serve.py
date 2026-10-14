import os, torch
from fastapi import FastAPI
from google.cloud import storage
from model import SwinXception

app = FastAPI()

def load_model():
    client = storage.Client()
    blob = client.bucket("swinxception").blob("models/swinxception-smoted/v1/swin_xception_final.pth")
    blob.download_to_filename("/tmp/model.pth")
    m = SwinXception()
    m.load_state_dict(torch.load("/tmp/model.pth", map_location="cpu", weights_only=True))
    m.eval()
    return m

model = load_model()

@app.post("/predict")
def predict(payload: dict):
    x = torch.tensor(payload["inputs"])
    with torch.no_grad():
        return {"outputs": model(x).tolist()}