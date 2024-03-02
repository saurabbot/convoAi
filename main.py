from fastapi import FastAPI, status
from fastapi.responses import JSONResponse


async def not_found(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "Not Found"},
    )


exception_handlers = {
    404: not_found,
}

app = FastAPI(exception_handlers=exception_handlers)


@app.get("/")
def root():
    return {"message": "Hello World"}
