from rag.retriever import RAGRetriever


def pretty_print_results(results):
    for i, item in enumerate(results, start=1):
        print(f"\n----- RESULT {i} -----")
        print("Score (distance):", item["score"])
        print("Source:", item["metadata"].get("source"))
        print("Metadata:", item["metadata"])
        print("Text:")
        print(item["text"])
        print("----------------------")


if __name__ == "__main__":
    # index_path should point to the directory where ChromaDB index lives
    rag = RAGRetriever(index_path="rag/index", collection_name="traffic_knowledge")

    query = "Ambulance priority rules on King Road"
    print("Query:", query)

    results = rag.retrieve(query, k=3)
    pretty_print_results(results)
