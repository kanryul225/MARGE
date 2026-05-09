## Overview

# Concept of Project

- clinical ML model orchestrating agent system, using tabular data-based clinical prediction ML model as a tool visible to agent.
- in User perspective, it is a clinical diagnosis assisting chat system. User can attach patients clinical data DB or personal clinical data. Agentic system can utilize them.
- Agent Structure and Division of roles:
  - main agent : orchestrator, professional ML researcher, helpful clinical assistant (in user perspective)
    - orchestrator agent doesn't do clinical prediction. it can use several ML models (XGBoost, CATboost, ... ) and derive XAI feature scores (such as SHAP score) of their predictions. orchestrator can also ask "Expert clinical diagnose agent" which is LLM trained on vast medical QA dataset.
    - orchestrator knows what each niche ML model is trained on(what they know), what input features they require and what they predict, what is their test accuracy (how reliable they are), and can utilize XAI scores of ML model's prediction(why they derived the prediction)
    - orchestrator is meant to choose niche ML models to get predictions, assess the reliabilities of each prediction, and organize, analyze, resolve conflicts, derive implications, and integrate numerous predictions for clinical diagnosis as a "Professional medical ML head researcher".
    - orchestrator can talk medical domain-expert agent, who is basically a chatbot trained on medical QA, working as a "Professional Doctor" in this system, to get a medical insight for analysis.
    - orchestrator SHOULD NOT make any arbitrary medical decision. It should use niche ML models to get data-driven medical analysis, and get confirmed by medical domain-expert agent to get medical insight.
    - orchestrator's goal is to put together several expert-ML model's numerical predictions in ML researcher perspective, and resolve them with medical domain-expert's knowledgable insight.
    - Another important role of orchestrator agent is Abstaintion and Asking back to user. If several ML model outputs give conflicting output that can't be resolved or if domain-expert's insight denote ML model predictions are not reliable or every ML models' outputs are unreliable, it should reply "I can not give you a reliable guidance in current knowledge. Give me extra A, B, C information so I can help you or I strongly recommend to visit hospital and see doctor for reliable diagnosis.". Orchestrator can ask user back for extra information that seems to help ML model's predictions and give a meaningful information gain.
  - medical domain-expert agent : professional doctor, retains bountiful knowledge on medical domain, can perform high-level medical reasoning
    - medical domain-expert agent is LLM trained on medical QA
    - it should be a source of medical insight.
    - main agent should inquire this agent to get a medical reasoning.
  - niche ML models : numerical predictor for clinical information
    - ML models are trained on various clinical tabular datasets
    - each model has fixed input features and output features
    - their trained dataset and test accuracy on the dataset are known
    - they offer predictions and XAI scores (such as SHAP index), indicating which features performed main roles making this prediction.
    - they are schematic API tool in respect of main agent (they can be exposed as a usable tool directly or through MCP server.)
    - input of the tool is ML model inputs
    - output of the tool is ML model prediction and XAI indices
  - clinical data retriever : Web/DB Search engine for clinical agent
    - medical domain-expert agent can search for medical knowledge from web or clinical Knowledge DB
    - orchestrator agent can retrieve patient clinical history from clinical history DB
    - Characteristics of information each agent can access differs clearly : medical domain-expert agent search for domain knowledge to help its clinical reasoning, orchestrator agent search for patient information for orchestration and next step decision.
    - this can be implemented as any form of attached extensions to system : Web search tool, DB, Filesystem, Search subagent, ... etc.

# Architectural Design Philosophy

We can use any open source framework or frontend/backend Structure to support following design.

- Main agent loop should be light. It should not contain heavy workload itself.
- Majority of features should be designed as a tool / harness / attched extension to agent for flexibility.
- Agents-ML models orchestration should be fast and flexible. (asychoronous predictions supporting)
- Orchestrator should always rely on ML models and expert agent to get a medical/clinical insight. It should never work as a chatbot that consumes medical questions

# Structural Diagram

```
┌──────────────────────────────────────────────────────┐
│  User (clinician + patient data attached)            │
└────────────────┬─────────────────────────────────────┘
                 │
        ┌────────▼─────────┐    "ML Head Researcher"
        │ Orchestrator     │    - light loop, no heavy work
        │ Agent            │    - never answers medically itself
        └─┬─────┬─────┬────┘    - resolves conflicts / abstains
          │     │     │
   ┌──────▼┐ ┌──▼──┐ ┌▼──────────────┐
   │ Niche │ │Med. │ │ Clinical Data │
   │ ML    │ │Expert│ │ Retriever    │
   │ Tools │ │Agent │ │ (Web/DB)     │
   │(XGB,  │ │ LLM  │ │              │
   │CATB+  │ │"Doc" │ │              │
   │SHAP)  │ │      │ │              │
   └───────┘ └──────┘ └──────────────┘
```

**Summary**

- **Orchestrator Agent** — the user-facing "ML head researcher". Runs a light, dynamic tool-use loop; selects which ML tools to call, integrates their predictions and XAI scores, consults the medical expert, and decides whether to answer, ask the user back, or abstain. Never produces medical claims from its own knowledge.
- **Niche ML Tools** — XGBoost / CATBoost / etc. trained on tabular clinical datasets. Each tool exposes a fixed input/output schema, the dataset it was trained on, its test accuracy, and SHAP-style XAI scores alongside every prediction. New models can be added without touching orchestrator code.
- **Medical Expert Agent** — an LLM trained on medical QA, acting as the "professional doctor". Provides clinical reasoning and confirms/refutes ML-derived signals. Has its own retriever scoped to medical knowledge sources.
- **Clinical Data Retrievers** — split by consumer: the orchestrator pulls _patient history_ (records, prior labs, demographics) for case context; the medical expert pulls _medical knowledge_ (literature, guidelines, KB) for reasoning. Same mechanism, distinct scopes.
- **Control behaviors** — conflict resolution across model outputs, ask-back when missing information would meaningfully shift predictions, and explicit abstention when no reliable answer exists. These are enforced structurally (tool design + hooks), not by prompting alone.
