from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from ultralytics import YOLO
import cv2
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "access_token"

app = FastAPI()
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

model = YOLO("yolo11s.pt")

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Acesso negado 🚫")
    return api_key


@app.post("/contar-produtos")
async def contar_produtos(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key) 
):
    # Salvar arquivo temporário
    contents = await file.read()
    arquivo_temp = f"/tmp/temp_{file.filename}"

    with open(arquivo_temp, "wb") as f:
        f.write(contents)

    print("PROCESSANDO IMAGEM...")

    img = cv2.imread(arquivo_temp)

    # Fazer inferência
    resultados = model(img, conf=0.4, iou=0.45)
    contador = len(resultados[0].boxes)
    anotada = resultados[0].plot()
    cv2.putText(anotada, f"Produtos: {contador}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    os.remove(arquivo_temp)

    return JSONResponse({"quantity": contador})


@app.get("/helloworld")
def obter_imagem(api_key: str = Depends(get_api_key)):
    return {"msg": "API FUNCIONANDO ✅"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)