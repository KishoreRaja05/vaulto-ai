"""Quick RAG retrieval check. Expected files are listed next to relevant queries."""

from app.services.rag_engine import rag

QUERIES = [
    ("what is an emergency fund?", "emergency-fund.md"),
    ("how should I split my salary?", "50-30-20-rule.md"),
    ("what is the 50/30/20 rule?", "50-30-20-rule.md"),
    ("how do I cut down eating out?", "reducing-food-spending.md"),
    ("how does the forecast work?", "how-vaulto-forecast-works.md"),
    ("how does anomaly detection work?", "how-vaulto-anomaly-detection-works.md"),
    ("how are transactions categorized?", "how-vaulto-categorization-works.md"),
    ("how do I budget with irregular income?", "irregular-income.md"),
    ("how do I reduce transport costs?", "reducing-transport-spending.md"),
    ("why is my food bill so high", "reducing-food-spending.md"),
    ("should I keep money for emergencies", "emergency-fund.md"),
    ("what is savings rate", "savings-rate.md"),
    ("how much should I save every month", "savings-rate.md"),
    ("how do I stop wasting money on subscriptions", "recurring-subscriptions.md"),
    ("how do I pay off my credit card fast", "high-interest-debt.md"),
    ("capital of France", None),
    ("best pizza recipe", None),
    ("how to cook rice", None),
    ("who won the cricket match", None),
    ("how do I fix my laptop?", None),
    ("what is the weather?", None),
]

threshold = rag.min_score
relevant_summary = []
offtopic_summary = []

print(f"RAG_MIN_SCORE = {threshold:.2f}")

for query, expected in QUERIES:
    results = rag.search(query, k=3, min_score=0.0)
    print("\n" + query)
    if not results:
        print("  (no results)")
    for result in results:
        passed = result["score"] >= threshold
        snippet = result["text"][:100].replace("\n", " ")
        print(f"  {result['score']:.3f}  {'PASS' if passed else 'FAIL'}  {result['source']}")
        print(f"      {snippet}")
    if expected:
        ok = any(
            result["source"] == expected and result["score"] >= threshold
            for result in results
        )
        relevant_summary.append((query, expected, ok))
    else:
        ok = not any(result["score"] >= threshold for result in results)
        offtopic_summary.append((query, ok))

print("\nPASS/FAIL SUMMARY")
for query, expected, ok in relevant_summary:
    print(f"  {'PASS' if ok else 'FAIL'}  {query} -> expected {expected}")
for query, ok in offtopic_summary:
    print(f"  {'PASS' if ok else 'FAIL'}  {query} -> expected no result above {threshold:.2f}")
print(
    f"Relevant: {sum(ok for _, _, ok in relevant_summary)}/{len(relevant_summary)} passed; "
    f"off-topic rejected: {sum(ok for _, ok in offtopic_summary)}/{len(offtopic_summary)} passed."
)
