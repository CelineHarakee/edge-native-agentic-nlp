from rag.retriever import RAGRetriever
from kg.kg_utils import TrafficKnowledgeGraph
from agents.incident_agent import IncidentDetectionAgent
from agents.traffic_light_agent import TrafficLightAgent


def run_pipeline(message: str, location_hint: str):
    print("\n===================================")
    print("VEHICLE / RAW MESSAGE:")
    print(message)

    rag = RAGRetriever(index_path="rag/index", collection_name="traffic_knowledge")
    kg = TrafficKnowledgeGraph()

    # Incident detection
    incident_agent = IncidentDetectionAgent(rag)
    incident_report = incident_agent.process_message(message)

    print("\nINCIDENT REPORT:")
    print("Severity:", incident_report["severity"])
    print("Incident type:", incident_report["incident_type"])

    # Traffic light decision
    traffic_agent = TrafficLightAgent(rag, kg)
    decision = traffic_agent.decide(incident_report, location_hint=location_hint)

    print("\nTRAFFIC LIGHT DECISION (structured):")
    print(decision["action"])
    print(decision["parameters"])

    print("\nEXPLANATION:")
    print(decision["explanation"])
    print("===================================\n")


if __name__ == "__main__":
    msg1 = "An ambulance is stuck in heavy congestion on Haram Road near the hospital, moving very slowly."
    msg2 = "There is a minor slow-down on Seaside Road due to light rain, but traffic is still flowing."

    run_pipeline(msg1, location_hint="Haram Road")
    run_pipeline(msg2, location_hint="Seaside Road")
