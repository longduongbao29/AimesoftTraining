import Rag.routers.api
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:fast_app", port=8080, reload=True)
