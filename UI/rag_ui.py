import gradio as gr
from matplotlib import scale
import requests
from pydantic import BaseModel, Field
from enum import Enum


# Define Pydantic Schemas
class ModeEnum(str, Enum):
    default = "default"
    multi_query = "multi-query"
    rag_fusion = "rag-fusion"
    recursive_decomposition = "recursive-decompostion"
    individual_decomposition = "individual-decomposition"
    step_back = "step-back"
    hyde = "hyde"


class Question(BaseModel):
    question: str = Field(examples=["What is your name?"])


class RetrieverSchema(BaseModel):
    mode: ModeEnum


class AskRequest(BaseModel):
    question: Question
    retrieval_schema: RetrieverSchema


# Define the retriever API endpoint
RETRIEVER_API_URL = "http://127.0.0.1:1111"


# Function to handle document upload
def upload_file(file):
    if file is not None:
        # Convert NamedString to BytesIO for file-like behavior
        with open(file, "rb") as f:
            response = requests.post(
                f"{RETRIEVER_API_URL}/upload",
                files={"file": (file.name, f, "text/plain")},
            )
            return response.json()
    return {"error": "No file uploaded"}


# Function to handle question and answer
def ask_question(history, question, mode):
    # Create request body
    request_body = AskRequest(
        question=Question(question=question),
        retrieval_schema=RetrieverSchema(mode=mode),
    )

    response = requests.post(f"{RETRIEVER_API_URL}/ask", json=request_body.dict())
    if response.status_code == 200:
        answer = response.json().get("output", "No answer found")
    else:
        answer = "Failed to retrieve answer"

    # Append the new question and answer to the chat history
    history.append((question, answer))
    return history


def clear_chat(history):
    # Clear the chat history
    return []


# Create Gradio interface
def create_app():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Chat with the Document")
                chatbot = gr.Chatbot(label="Chat Interface")
                with gr.Row():
                    question_input = gr.Textbox(label="Ask a question", scale=5)
                    send_button = gr.Button("Send", scale=1)
                mode_input = gr.Dropdown(
                    label="Select Mode",
                    choices=[mode.value for mode in ModeEnum],
                    value="default",
                )

                clear_button = gr.Button("Clear Chat")

                # The `state` argument keeps track of the chat history
                send_button.click(
                    ask_question,
                    inputs=[chatbot, question_input, mode_input],
                    outputs=chatbot,
                )
                clear_button.click(clear_chat, inputs=chatbot, outputs=chatbot)
            with gr.Column(scale=1):
                gr.Markdown("## Upload Document")
                file_input = gr.File(
                    label="Upload your document", file_types=[".pdf", ".txt", ".docx"]
                )
                upload_button = gr.Button("Upload")
                upload_output = gr.Textbox(label="Upload Status")
                upload_button.click(
                    upload_file, inputs=file_input, outputs=upload_output
                )
    return demo


# Launch the app
ui_app = create_app()
ui_app.launch(server_port=1234, share=True)  # Set share=True to create a public link
