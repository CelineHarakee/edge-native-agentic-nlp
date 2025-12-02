from typing import Dict, Any, Optional, List

from rag.retriever import RAGRetriever
from kg.kg_utils import TrafficKnowledgeGraph
from llm.client import call_llm


class VehicleAgent:
    """
    VehicleAgent is responsible for:
    - Observing local sensor data (speed, road, vehicle type, congestion, etc.)
    - Querying RAG to retrieve relevant rules/incidents for the current road
    - Generating a natural-language description of the current situation
    """

    def __init__(
        self,
        rag: RAGRetriever,
        kg: Optional[TrafficKnowledgeGraph] = None,
    ) -> None:
        self.rag = rag
        self.kg = kg
        self.last_observation: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # Core methods
    # --------------------------------------------------

    def observe(self, sensor_data: Dict[str, Any]) -> None:
        """
        Store the latest sensor reading.

        Example sensor_data:
            {
                "speed": 12,
                "road": "King Road",
                "vehicle_type": "ambulance",
                "congestion_level": "high"
            }
        """
        self.last_observation = sensor_data

    def _build_rag_query(self, sensor_data: Dict[str, Any]) -> str:
        """
        Build a simple text query for RAG based on the sensor data.
        """
        road = sensor_data.get("road", "")
        vehicle_type = sensor_data.get("vehicle_type", "vehicle")
        congestion = sensor_data.get("congestion_level", "unknown congestion")
        speed = sensor_data.get("speed", None)

        base_query = (
            f"traffic rules and incidents related to {vehicle_type} on {road} "
            f"with {congestion} congestion"
        )

        if speed is not None:
            base_query += f" and speed around {speed} km/h"

        return base_query

    def _format_context(self, rag_results: List[dict]) -> str:
        """
        Turn RAG results into a compact bullet-style context string.
        """
        if not rag_results:
            return "No additional context found."

        lines = []
        for item in rag_results:
            src = item["metadata"].get("source", "unknown_source")
            text = item["text"]
            lines.append(f"- Source: {src}. Info: {text}")

        return "\n".join(lines)

    def generate_message(self, sensor_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Use the latest observation (or provided sensor_data) plus RAG context
        to generate a natural-language message describing the situation.
        """
        if sensor_data is not None:
            self.observe(sensor_data)

        if self.last_observation is None:
            return "[VehicleAgent] No observation available."

        obs = self.last_observation

        # 1) Build RAG query
        rag_query = self._build_rag_query(obs)

        # 2) Retrieve context from RAG
        rag_results = self.rag.retrieve(rag_query, k=3)
        context_str = self._format_context(rag_results)

        # 3) Optionally, use KG for extra info (e.g., nearest hospital)
        kg_note = ""
        if self.kg is not None and "road" in obs:
            road_name = obs["road"]
            # Try to see if the road is in the graph as a label
            neighbors = self.kg.get_neighbors(road_name)
            if neighbors:
                kg_note = (
                    f"The road {road_name} is connected to: "
                    + ", ".join(neighbors[:3])
                    + "."
                )

        # 4) Build prompt for Gemini
        prompt = f"""
You are the Vehicle Agent in a smart city traffic coordination system.
Your job is to describe the current local traffic situation clearly and briefly
so that other agents (Incident Detection Agent and Traffic Light Agent) can use it.

Current sensor data (JSON):
{obs}

Retrieved context from the knowledge base:
{context_str}

Additional graph-based note (if any):
{kg_note}

Please:
- Summarize the current situation in 2–3 sentences.
- Explicitly mention the road name, vehicle type, congestion level, and whether this looks risky or normal.
- Be factual and concise, no extra explanations.
"""

        message = call_llm(prompt)
        return message
