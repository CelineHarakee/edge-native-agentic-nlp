from rag.retriever import RAGRetriever
from kg.kg_utils import TrafficKnowledgeGraph
from agents.vehicle_agent import VehicleAgent
from agents.incident_agent import IncidentDetectionAgent
from agents.traffic_light_agent import TrafficLightAgent


def run_scenario(name: str, sensor_data: dict):
    print("\n" + "=" * 50)
    print(f"SCENARIO: {name}")
    print("=" * 50)

    # 1) Init shared components
    rag = RAGRetriever(index_path="rag/index", collection_name="traffic_knowledge")
    kg = TrafficKnowledgeGraph()

    # 2) Init agents
    vehicle_agent = VehicleAgent(rag=rag, kg=kg)
    incident_agent = IncidentDetectionAgent(rag=rag)
    traffic_agent = TrafficLightAgent(rag=rag, kg=kg)

    # 3) VehicleAgent: observe + describe
    print("\n[1] VEHICLE AGENT – SENSOR DATA")
    print(sensor_data)

    vehicle_msg = vehicle_agent.generate_message(sensor_data)
    print("\n[1] VEHICLE AGENT – MESSAGE")
    print(vehicle_msg)

    # 4) IncidentDetectionAgent: classify + retrieve similar cases
    incident_report = incident_agent.process_message(vehicle_msg)

    print("\n[2] INCIDENT AGENT – STRUCTURED REPORT")
    print("Severity:", incident_report["severity"])
    print("Incident type:", incident_report["incident_type"])
    print("Similar cases:")
    for case in incident_report["similar_cases"]:
        print(
            f"  - Road: {case['road_name']}, "
            f"Severity: {case['severity_label']}, "
            f"Score: {case['score']:.3f}"
        )

    print("\n[2] INCIDENT AGENT – NARRATIVE REPORT")
    print(incident_report["report"])

    # Pick a location_hint for TrafficLightAgent
    location_hint = sensor_data.get("road")

    # 5) TrafficLightAgent: decide action
    decision = traffic_agent.decide(
        incident_report=incident_report,
        location_hint=location_hint,
    )

    print("\n[3] TRAFFIC LIGHT AGENT – DECISION (STRUCTURED)")
    print("Action:", decision["action"])
    print("Parameters:", decision["parameters"])

    print("\n[3] TRAFFIC LIGHT AGENT – EXPLANATION")
    print(decision["explanation"])

    print("\n" + "=" * 50)
    print(f"END OF SCENARIO: {name}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Scenario 1: Critical ambulance delay
    scenario1_sensor = {
        "speed": 8,
        "road": "Haram Road",
        "vehicle_type": "ambulance",
        "congestion_level": "high",
    }

    # Scenario 2: Mild congestion due to weather
    scenario2_sensor = {
        "speed": 35,
        "road": "Seaside Road",
        "vehicle_type": "private_car",
        "congestion_level": "medium",
    }

    run_scenario("Ambulance stuck in congestion near hospital", scenario1_sensor)
    run_scenario("Minor slow-down due to rain", scenario2_sensor)
