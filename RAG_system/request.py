import requests

url = "http://localhost:8000/query"
question = {"question": "What is the scope of regulation of road traffic law?"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=question, headers=headers)
print(response.json())
