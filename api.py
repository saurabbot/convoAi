from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import pymongo
import sys

app = FastAPI()

try:
    client = client = pymongo.MongoClient(
        "mongodb+srv://firefox:ml99C2ne3bMaIaAQ@cluster0.1owkict.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    )

except pymongo.errors.ConfigurationError:
    print(
        "An Invalid URI host error was received. Is your Atlas host name correct in your connection string?"
    )
    sys.exit(1)


@app.get("/")
def root():
    return {"message": "Hello sorld"}
