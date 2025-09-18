from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="KRN MRP Plant1")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Routers import
from .routers import grn, heat, qa, lot
app.include_router(grn.router)
app.include_router(heat.router)
app.include_router(qa.router)
app.include_router(lot.router)

# Root test route
@app.get("/")
def read_root():
    return {"msg": "KRN MRP Plant1 – Render/Neon Ready with Full Workflow"}
