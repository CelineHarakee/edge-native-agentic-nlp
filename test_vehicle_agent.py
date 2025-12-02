from rag.retriever import RAGRetriever
from kg.kg_utils import TrafficKnowledgeGraph
from agents.vehicle_agent import VehicleAgent


if __name__ == "__main__":
    # Initialize RAG and KG
    rag = RAGRetriever(index_path="rag/index", collection_name="traffic_knowledge")
    kg = TrafficKnowledgeGraph()  # uses data/graph_data.json by default

    # Create the VehicleAgent
    vehicle_agent = VehicleAgent(rag=rag, kg=kg)

    # Example sensor data
    sensor_data = {
        "speed": 12,
        "road": "Haram Road",
        "vehicle_type": "ambulance",
        "congestion_level": "high"
    }

    print("=== SENSOR DATA ===")
    print(sensor_data)

    print("\n=== VEHICLE AGENT MESSAGE ===")
    message = vehicle_agent.generate_message(sensor_data)
    print(message)
