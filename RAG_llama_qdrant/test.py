from llama_cpp import Llama

llm = Llama(
    model_path="D:\Code\AimesoftTraining\RAG_llama_qdrant\models\llama-2-7b-chat.Q2_K.gguf",
    chat_format="chatml",
)
out = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",
    },
    temperature=0.7,
)
print(out)