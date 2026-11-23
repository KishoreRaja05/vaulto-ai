# How Vaulto Categorization Works

How does Vaulto categorise transactions? Vaulto turns noisy bank descriptions into structured categories with a trained text classifier.

## The problem

Bank descriptions are noisy. Examples:

- POS 4410 SWIGGY BANGALORE IN
- UPI/412398765432/ZOMATO/zomato@hdfcbank/Payment
- UPI-NETFLIX-netflix@icici
- NACH DR SIP HDFC MF
- APL* ITUNES.COM/BILL CA

## How the classifier works

The transaction description is vectorised with TF-IDF over words and character n-grams, then a trained Logistic Regression model predicts the category with about 0.88 accuracy on the current dataset. This figure measures how well the model imitates keyword rules on a synthetic dataset, not real-world accuracy. The result includes the predicted category and a confidence score.

Common merchants such as Swiggy, Zomato, Netflix and Uber appear in the training dataset, where some labels were assigned using keyword rules, so the model mainly reproduces those patterns.

Person-to-person UPI transfers often need user labelling, since the narration rarely says what the payment was for. If the model weights are not loaded, the prediction falls back to "Other" with zero confidence.
