# Semantic Multi-Agent Traffic Coordination System (NLP + RAG + KG + LLM)
A simulation of a smart, edge-native traffic coordination system where autonomous agents communicate using natural language, enriched with:
- NLP processing
- Retrieval-Augmented Generation (RAG)
- Knowledge Graph reasoning (KG)
- LLM-based decision making
- Synthetic sensor data from edge devices
- Interactive Streamlit dashboard
  
This project demonstrates how agents can use semantic (text-based) communication to improve interpretability and coordination in smart-city traffic systems.

## Project Overview
The system contains three cooperative agents:
1. **VehicleAgent**
  - Represents an edge IoT vehicle device
  - Receives sensor data (speed, road, weather, congestion, vehicle type)
  - Uses RAG retrieval + LLM summarization
  - Outputs a natural-language situation report
    
2. **IncidentDetectionAgent**
  - Reads the VehicleAgent’s message
  - Uses LLM for zero-shot incident classification
  - Retrieves similar cases via RAG
  - Produces a structured report:
    
    ` {
  "severity": "...",
  "incident_type": "...",
  "report": "...",
  "similar_cases": [...]
  } `

3. **TrafficLightAgent**
- Reads the incident report
- Uses KG to find alternative routes / hospital paths
- Uses RAG for relevant policies
- Uses LLM to decide:
  - extend green
  - route ambulance
  - rebalance timing
  - normal monitoring
- Returns a final decision + natural-language explanation

## Core AI Components
### NLP Pipeline (Preprocessing)
- Cleaning, tokenization
- Stopword removal (domain-adjusted)
- Lemmatization
- Named Entity Recognition (NER)
- Bigram & trigram detection
- Vectorization (BoW, TF-IDF, TF-IDF N-gram)
- Classical ML models for severity classification
- Evaluation (accuracy, classification report)
  
### Retrieval-Augmented Generation (RAG)
Two modes:
1. Full embedding-based RAG (sentence-transformers + ChromaDB) for VS Code
2. Lightweight keyword-based RAG for Streamlit demo (no heavy deps)

### Knowledge Graph
Built with NetworkX using:
- nodes: roads, intersections, hospitals
- edges: CONNECTED_TO, NEAR, LEADS_TO
- utils:
  - get_neighbors(node)
  - find_path(source, target)
  - nearest_hospital(road)

## Setup Instructions
1. Clone the repo
` git clone https://github.com/yourusername/nlp-project.git `
` cd nlp-project ` 
2. Create virtual environment
`python3 -m venv .venv`
`source .venv/bin/activate`
3. Install dependencies
`pip install -r requirements.txt`
If you’re using Gemini:
`pip install google-generativeai`
4. Add your API key
(Create .env):
`GEMINI_API_KEY=your_key_here`
5. Run the Full Agent Simulation
`python main.py`


## NLP Evaluation Results
Classical NLP models were trained on synthetic incident text:
- Logistic Regression (TF-IDF): 0.80 accuracy
- SVM (TF-IDF): 0.80 accuracy
- Naive Bayes: ~0.60 accuracy
These results demonstrate that:
- incident severity is learnable from textual descriptions
- NLP classification can effectively support semantic agents
  
## Contributers
- [Celine Al Harake](https://github.com/CelineHarakee)
- [Layal Canoe](https://github.com/layalcanoe)
- Judy Abu Quta
