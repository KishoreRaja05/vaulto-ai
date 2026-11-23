# How Vaulto Forecast Works

How does Vaulto forecast monthly spending? Vaulto projects month-end spend from your transactions so far using a linear trend on cumulative daily spending.

## How the forecast is calculated

With at least 30 transactions and two complete months of history, Vaulto fits a linear regression to cumulative spend by day of the month and uses it to estimate spend at month end. It reports the fit as R² and gives a 90% residual interval so you can see how uncertain the estimate is. Confidence is high when there are at least six months of history and R² is at least 0.7, otherwise it is medium.

With less history, Vaulto uses a rule-based projection: regular spending is extrapolated from the daily rate, and one-off expenses are added back as a fixed amount instead of being extrapolated. Confidence is low and no R² or interval is produced.

## What is included in the result

The forecast reports projected month-end spend, average daily spend, the days elapsed and remaining in the month, the top spending category, per-category averages, confidence, the method used ("ml" or "rule_based") and short insights. One-off expenses that were excluded are listed separately.

## Reading a forecast

For example, if ₹8,400 is already spent against a ₹12,000 budget with 18 days left, you could aim for about ₹200 a day to stay within budget.
