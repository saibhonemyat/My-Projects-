from fastapi import FastAPI

app = FastAPI()

@app.get("/api")
def home():
    return {"status": "it's alive"}
