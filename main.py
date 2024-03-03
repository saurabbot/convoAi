from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
# from langchain.llms.openai import OpenAI
from src.utils.data_ingester import ingest_data

ingest_data()


# llm = OpenAI(api_key="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")
# output = llm.invoke("Whats up bitch?")
# print(output)
# async def not_found(request, exc):
#     return JSONResponse(
#         status_code=status.HTTP_404_NOT_FOUND,
#         content={"message": "Not Found"},
#     )


# exception_handlers = {
#     404: not_found,
# }
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}
