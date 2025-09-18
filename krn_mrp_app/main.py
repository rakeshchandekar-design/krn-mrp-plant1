
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "KRN MRP Plant1 – Render/Neon Ready"}
