from llm.client import call_llm

if __name__ == "__main__":
    prompt = "Summarize this situation in one sentence: An ambulance is stuck in heavy traffic near Intersection 4 on King Road."
    result = call_llm(prompt)
    print("Model response:")
    print(result)
