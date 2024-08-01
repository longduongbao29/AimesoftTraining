import main
question = "Main road means a road ensuring major traffic in an area."
result = main.qa_chain({"query": question})

print(result)
