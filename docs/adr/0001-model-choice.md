# ADR 0001: Model choice for the first production model

## Status

Accepted.

## Context

The service predicts 30-day readmission from the UCI diabetes encounters
dataset. Three model families were trained on identical features and the
same patient-grouped split: logistic regression, histogram gradient
boosting, and LightGBM. The winner is picked by ROC AUC on the held-out
patients, and every model's result is recorded in artifacts/metrics.json.

## Decision

Ship whichever model wins on ROC AUC, and keep the runner-up's metrics in
the model card. The serving layer returns exact per-feature contributions
when the winner is logistic regression. When the winner is the boosting
model, the API returns global permutation importances from the model card
and labels them as global. It does not fake per-request explanations.

## Trade-offs

Logistic regression is the simpler production model: cheap to retrain,
easy to explain to a clinician, and its coefficients are the explanation.
The two boosting models capture interactions the linear model misses
(prior utilization against medication count, for example) at the cost of
weaker explanations and a larger artifact. Between the boosters,
LightGBM trains faster on wide one-hot features, while the histogram
gradient boosting model ships inside scikit-learn with no extra
dependency. That operational difference only matters if their scores are
close.

For a first production model the deciding factor is measured lift, not
fashion. If the AUC gap between the two models were small (under about
0.01), the right call would flip to logistic regression for the simpler
operations story. The gap measured in this run is recorded in
artifacts/metrics.json and in the README, so the choice can be revisited
with data rather than opinion.

## Consequences

- Retraining must rerun the comparison, not just refit the winner.
- The API response carries an attribution_method field so clients know
  which kind of explanation they received.
- A future step is probability calibration, which matters more than a
  third model family once risk bands are shown in a product.

## Outcome of the first run

Measured 2026-10-15 and recorded in artifacts/metrics.json: LightGBM
won at ROC AUC 0.670, ahead of histogram gradient boosting at 0.6685 and
logistic regression at 0.667. The gap to logistic regression is 0.003,
inside the 0.01 band named above where the simpler model would also be
defensible. LightGBM shipped anyway, per the decision rule (ship the
AUC winner). The cost is the one the ADR already accepted: explanations
are global permutation importances, labelled as global in every API
response. If a product surface ever needs per-patient explanations,
retraining reruns the comparison and logistic regression is the
fallback. The decision threshold for the shipped model is a separate
decision, recorded in ADR 0002.
