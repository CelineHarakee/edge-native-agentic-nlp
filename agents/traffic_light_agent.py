from typing import Dict, Any, List, Optional

from rag.retriever import RAGRetriever
from kg.kg_utils import TrafficKnowledgeGraph
from llm.client import call_llm


class TrafficLightAgent:
    """
    TrafficLightAgent:
    - Receives a structured incident_report from IncidentDetectionAgent
    - Uses RAG to retrieve relevant policies and protocols
    - Uses the Knowledge Graph to reason about alternate routes / hospitals
    - Chooses a structured signal-control action
    - Generates a natural-language explanation of the decision
    """

    def __init__(
        self,
        rag: RAGRetriever,
        kg: TrafficKnowledgeGraph,
    ) -> None:
        self.rag = rag
        self.kg = kg

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def decide(
        self,
        incident_report: Dict[str, Any],
        location_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        incident_report example (from IncidentDetectionAgent):
        {
            "severity": "high",
            "incident_type": "emergency_delay",
            "similar_cases": [...],
            "report": "..."
        }

        location_hint: road name / intersection (e.g., "Haram Road")
        """
        severity = incident_report.get("severity", "medium")
        incident_type = incident_report.get("incident_type", "other")

        # Try to guess road_name from similar_cases if not provided
        road_name = location_hint
        if road_name is None:
            sims = incident_report.get("similar_cases") or []
            if sims:
                road_name = sims[0].get("road_name")

        # 1) Retrieve relevant policies / protocols via RAG
        rag_context = self._fetch_policy_context(severity, incident_type, road_name)

        # 2) Use KG to find alternate routes / hospitals (if needed)
        graph_info = self._compute_graph_context(road_name, severity, incident_type)

        # 3) Choose a structured action (simple rule-based logic)
        decision = self._choose_action(severity, incident_type, road_name, graph_info)

        # 4) Ask the LLM to explain the decision
        explanation = self._generate_explanation(
            incident_report=incident_report,
            decision=decision,
            rag_context=rag_context,
            graph_info=graph_info,
        )

        decision["explanation"] = explanation
        return decision

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _fetch_policy_context(
        self,
        severity: str,
        incident_type: str,
        road_name: Optional[str],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Use RAG to retrieve emergency protocols and traffic policies
        relevant to the incident.
        """
        query = (
            f"Traffic signal and emergency handling policies for severity {severity} "
            f"and incident type {incident_type}."
        )
        if road_name:
            query += f" Road: {road_name}."

        results = self.rag.retrieve(query, k=k)

        # We care mainly about policies and emergency protocols
        filtered = []
        for item in results:
            src = item["metadata"].get("source")
            if src in {"traffic_policy", "emergency_protocol", "road_rule"}:
                filtered.append(item)

        return filtered

    def _compute_graph_context(
        self,
        road_name: Optional[str],
        severity: str,
        incident_type: str,
    ) -> Dict[str, Any]:
        """
        Use KG to find neighbors / potential routes / nearest hospital.
        """
        info: Dict[str, Any] = {
            "nearest_hospital": None,
            "path_to_hospital": None,
            "neighbors": None,
        }

        if not road_name:
            return info

        # Neighbors of the road node (could be intersections etc.)
        neighbors = self.kg.get_neighbors(road_name)
        info["neighbors"] = neighbors

        # For serious incidents, we care about nearest hospital
        if severity in {"high", "critical"} or incident_type in {
            "emergency_delay",
            "multi_vehicle_collision",
            "fire",
        }:
            # Heuristic: try each neighbor as a starting intersection
            best_target = None
            best_path = None

            for nb in neighbors:
                result = self.kg.find_nearest_node_of_type(nb, "hospital")
                if not result:
                    continue
                hosp_label, path_labels = result
                if best_path is None or len(path_labels) < len(best_path):
                    best_target = hosp_label
                    best_path = path_labels

            if best_target and best_path:
                info["nearest_hospital"] = best_target
                info["path_to_hospital"] = best_path

        return info

    def _choose_action(
        self,
        severity: str,
        incident_type: str,
        road_name: Optional[str],
        graph_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Very simple rule-based policy for now.
        """
        action: str
        params: Dict[str, Any] = {}

        if severity in {"high", "critical"} and incident_type in {
            "emergency_delay",
            "fire",
            "multi_vehicle_collision",
        }:
            action = "give_emergency_priority"
            params["extend_green_seconds"] = 30
            params["target_road"] = road_name
            if graph_info.get("path_to_hospital"):
                params["priority_route"] = graph_info["path_to_hospital"]

        elif severity in {"medium", "high"} and incident_type in {
            "congestion",
            "event_crowd",
            "roadwork",
        }:
            action = "rebalance_signal_timing"
            params["extend_green_seconds"] = 15
            params["target_road"] = road_name
            params["note"] = "favor congested approach and monitor queues"

        elif severity == "low":
            action = "monitor_and_log"
            params["target_road"] = road_name
            params["note"] = "no strong intervention required, just observation"

        else:
            action = "default_safe_operation"
            params["target_road"] = road_name
            params["note"] = "keep normal signal plan with minor adjustments"

        return {
            "action": action,
            "parameters": params,
        }

    def _generate_explanation(
        self,
        incident_report: Dict[str, Any],
        decision: Dict[str, Any],
        rag_context: List[Dict[str, Any]],
        graph_info: Dict[str, Any],
    ) -> str:
        """
        Ask LLM to produce a concise explanation, using all the context.
        """
        severity = incident_report.get("severity")
        incident_type = incident_report.get("incident_type")
        vehicle_summary = incident_report.get("report")

        # Flatten RAG context
        policy_lines = []
        for item in rag_context:
            src = item["metadata"].get("source")
            text = item["text"]
            policy_lines.append(f"- Source: {src}. Info: {text}")
        policy_context = "\n".join(policy_lines) if policy_lines else "No specific policies found."

        explanation_prompt = f"""
You are the Traffic Light Agent in a smart city multi-agent system.

Incident classification from the IncidentDetectionAgent:
- Severity: {severity}
- Incident type: {incident_type}

VehicleAgent / Incident description:
{vehicle_summary}

Relevant policies and protocols from the knowledge base:
{policy_context}

Graph-based reasoning:
Nearest hospital: {graph_info.get('nearest_hospital')}
Path to hospital: {graph_info.get('path_to_hospital')}
Neighbors of affected road: {graph_info.get('neighbors')}

Chosen signal control decision (structured):
{decision}

Explain in 3–4 concise sentences:
- Why this action is appropriate for the severity and incident type.
- How it helps traffic flow and/or emergency response.
- Mention any use of alternate routes or hospital access if present.
Be factual, no extra fluff, no JSON.
"""

        return call_llm(explanation_prompt).strip()
