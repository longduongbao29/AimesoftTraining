FROM python:3.9-slim

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

ARG PORT=7000 
ENV PORT=$PORT

EXPOSE $PORT
COPY  /home/aime/.cache/huggingface/hub/models--tsmatz--xlm-roberta-ner-japanese/snapshots/39c525b11d35426fea754e906c4847208582647f /app/model

CMD ["sh", "-c", "uvicorn service:app --host 0.0.0.0 --port $PORT"]
