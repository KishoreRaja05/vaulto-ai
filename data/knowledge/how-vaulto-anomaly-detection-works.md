# How Vaulto Anomaly Detection Works

How does Vaulto detect unusual spending? Vaulto uses Isolation Forest to flag transactions whose amounts are unusual compared with your normal spending.

## Algorithm: Isolation Forest

Isolation Forest suits financial transaction data because unusual transactions are separated from the rest with fewer decision-tree splits than normal ones. Normal transactions form a cluster that needs many splits to isolate a single point, while an anomalous transaction is isolated in few splits, close to the root.

For example, if your usual grocery spend is ₹800-₹2,500, a one-off ₹18,500 charge at an unfamiliar merchant late at night would be a candidate for review.

## What the model uses

Vaulto computes the log of each expense amount and scores it with a trained Isolation Forest model. The caller supplies the contamination level, clipped between 0.001 and 0.5, which controls how sensitive the flagging is. When the model has recorded training scores, the threshold is taken from the score percentile for that contamination level; otherwise the model's own prediction is used.

## What happens to flagged transactions

Flagged items are shown to the user for review, for example: "This ₹18,500 payment is much higher than your usual spend in Shopping." Vaulto flags unusual activity but does not block payments itself.
