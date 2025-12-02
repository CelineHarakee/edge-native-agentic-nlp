from rag.retriever import RAGRetriever
from agents.incident_agent import IncidentDetectionAgent


def run_example(vehicle_message: str):
    print("\n==============================")
    print("VEHICLE MESSAGE:")
    print(vehicle_message)

    rag = RAGRetriever(index_path="rag/index", collection_name="traffic_knowledge")
    agent = IncidentDetectionAgent(rag=rag)

    result = agent.process_message(vehicle_message)

    print("\nSTRUCTURED RESULT:")
    print("Severity:", result["severity"])
    print("Incident type:", result["incident_type"])
    print("Similar cases (up to 3):")
    for case in result["similar_cases"]:
        print(f"  - Road: {case['road_name']}, "
              f"Severity: {case['severity_label']}, "
              f"Score: {case['score']:.3f}")

    print("\nREPORT:")
    print(result["report"])
    print("==============================\n")


if __name__ == "__main__":
    msg1 = "An ambulance is stuck in heavy congestion on Haram Road near the hospital, moving very slowly."
    msg2 = "There is a minor slow-down on Seaside Road due to light rain, but traffic is still flowing."

    run_example(msg1)
    run_example(msg2)
