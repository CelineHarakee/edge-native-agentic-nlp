import json
from typing import Dict, Any, List

from rag.retriever import RAGRetriever
from llm.client import call_llm


class IncidentDetectionAgent:
    """
    IncidentDetectionAgent:
    - Receives a natural-language message from the VehicleAgent
    - Uses the LLM to classify severity + incident type (zero-shot NLP)
    - Uses RAG to retrieve similar historical incident cases
    - Returns a structured incident summary + explanation text
    """

    def __init__(self, rag: RAGRetriever) -> None:
        self.rag = rag

        # Allowed labels (match your dataset: incident_cases.json)
        self.allowed_severity = ["low", "medium", "high", "critical"]
        self.allowed_incident_types = [
            "rear_end_collision",
            "multi_vehicle_collision",
            "breakdown",
            "congestion",
            "event_crowd",
            "roadwork",
            "flooding",
            "pedestrian_crossing",
            "fire",
            "emergency_delay",
            "other"
        ]

    # --------------------------------------------------
    # Core public API
    # --------------------------------------------------

    def process_message(self, vehicle_message: str) -> Dict[str, Any]:
        """
        Main entry point.

        Input:
            vehicle_message: natural-language description from VehicleAgent

        Output (example dict):
        {
            "severity": "high",
            "incident_type": "emergency_delay",
            "similar_cases": [...],
            "report": "natural language explanation..."
        }
        """
        # 1) Zero-shot classification via LLM
        classification = self._classify_incident(vehicle_message)

        severity = classification.get("severity", "unknown")
        incident_type = classification.get("incident_type", "other")

        # 2) Retrieve similar incidents from RAG
        similar_cases = self._retrieve_similar_cases(
            vehicle_message,
            severity=severity,
            incident_type=incident_type,
            k=3,
        )

        # 3) Ask LLM to generate a concise human-readable report
        report = self._generate_report(
            vehicle_message,
            severity,
            incident_type,
            similar_cases,
        )

        return {
            "severity": severity,
            "incident_type": incident_type,
            "similar_cases": similar_cases,
            "report": report,
        }

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _classify_incident(self, vehicle_message: str) -> Dict[str, str]:
        """
        Ask the LLM to do zero-shot classification and return a JSON dict
        with fields: severity, incident_type.
        """
        system_instruction = (
            "You are an incident detection module in a traffic control system. "
            "You must classify the severity and incident type using fixed labels."
        )

        prompt = f"""
You will receive a short description of a traffic situation.

You MUST respond with ONLY a JSON object, no extra text, using this schema:

{{
  "severity": "low|medium|high|critical",
  "incident_type": "rear_end_collision|multi_vehicle_collision|breakdown|congestion|event_crowd|roadwork|flooding|pedestrian_crossing|fire|emergency_delay|other"
}}

Rules:
- severity reflects how urgent/dangerous the situation is.
- incident_type is the best matching cause/type from the list.
- Do not add comments, explanations or extra fields.

Traffic situation:
\"\"\"{vehicle_message}\"\"\""""

        raw = call_llm(prompt, system_instruction=system_instruction)

        # Try to parse JSON safely
        try:
            # Sometimes the model might wrap JSON in markdown; strip code fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                # remove possible language hint like json\n
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]
            data = json.loads(cleaned)
        except Exception:
            # Fallback if parsing fails
            data = {
                "severity": "medium",
                "incident_type": "other",
            }

        # Enforce valid labels
        sev = data.get("severity", "medium").lower()
        if sev not in self.allowed_severity:
            sev = "medium"

        itype = data.get("incident_type", "other")
        if itype not in self.allowed_incident_types:
            itype = "other"

        return {
            "severity": sev,
            "incident_type": itype,
        }

    def _retrieve_similar_cases(
        self,
        vehicle_message: str,
        severity: str,
        incident_type: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Use RAG to fetch similar incident cases based on the message and labels.
        Returns a list of simplified dicts for the top-k results.
        """
        query = (
            f"Similar incident cases with severity {severity} "
            f"and type {incident_type}. Situation: {vehicle_message}"
        )

        results = self.rag.retrieve(query, k=k)

        simplified = []
        for item in results:
            meta = item.get("metadata", {})
            if meta.get("source") != "incident_case":
                # We only keep actual incident_case entries here
                continue

            simplified.append(
                {
                    "road_name": meta.get("road_name"),
                    "severity_label": meta.get("severity_label"),
                    "row_id": meta.get("row_id"),
                    "text": item.get("text"),
                    "score": item.get("score"),
                }
            )

        return simplified

    def _generate_report(
        self,
        vehicle_message: str,
        severity: str,
        incident_type: str,
        similar_cases: List[Dict[str, Any]],
    ) -> str:
        """
        Ask the LLM to produce a short explanation of the incident, using
        the classification + similar cases as context.
        """
        similar_text = ""
        for case in similar_cases:
            similar_text += (
                f"- Road: {case.get('road_name')}, "
                f"Severity: {case.get('severity_label')}, "
                f"Details: {case.get('text')}\n"
            )

        prompt = f"""
You are the Incident Detection Agent in a smart city traffic system.

You have classified the incident as:
- Severity: {severity}
- Incident type: {incident_type}

Original message from Vehicle Agent:
\"\"\"{vehicle_message}\"\"\"


Similar past incident cases:
{similar_text if similar_text else "No close historical cases found."}

Write a short report (3–4 sentences) that:
- Clearly states the severity and incident type.
- Briefly explains why this classification makes sense.
- Mentions if this could impact emergency response or traffic flow.
Be concise and factual, no extra fluff.
"""

        report = call_llm(prompt)
        return report.strip()
