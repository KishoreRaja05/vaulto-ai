# Vaulto FAQ

What is Vaulto? Vaulto is an AI-powered personal finance tracker that combines expense tracking, machine learning models, a RAG knowledge base of financial guidance and a natural-language chat assistant in one platform. Amounts are shown in ₹ (INR).

## What is Vaulto?

- ML models run inside the FastAPI service boundary.

## How is my data handled?

Your transactions are stored in Vaulto's database and are only returned to you
after your login token is checked. When you use the chat, a hosted language
model (Groq) receives your question and the results of Vaulto's tools, such as
spending totals by category or forecast figures, so it can write an answer.

## How does the chat work?

What is the difference between Vaulto's ML engine and its LLM?

- ML engine (Python / FastAPI): numerical work such as spending forecasts, anomaly detection and transaction categorisation.
- LLM engine (Groq / RAG): understands your questions, retrieves financial knowledge from the RAG store, interprets context and calls backend APIs through structured tools.

The assistant can add transactions through Vaulto's backend, which checks your login. It does not have direct database access.

## Example questions

- "Am I on track to stay within my ₹40,000 monthly budget?"
- "Explain the 50/30/20 rule and apply it to my ₹60,000 in-hand salary."
- "Show me any unusual charges flagged this week."
- "How big should my emergency fund be if my monthly essentials are ₹30,000?"
