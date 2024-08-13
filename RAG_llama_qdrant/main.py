from init import fast_app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(fast_app, port=8080, reload=True)
