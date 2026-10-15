# ADR 0002: Decision threshold for the high risk flag

## Status

Accepted.

## Context

The first training run scored the winning LightGBM model at the default
0.5 threshold: precision 0.584, recall 0.027 on the held-out test
patients. The model ranks patients reasonably (ROC AUC 0.670) but puts
almost no probability mass above 0.5, because only 11.19 percent of
encounters are readmissions. A flag that misses 97 percent of the
patients who actually come back is not a screening tool, whatever its
precision says.

## Decision

The API keeps returning the raw probability. The boolean high_risk flag
is cut at a threshold tuned at training time: a patient-grouped
validation slice (20 percent of the training rows) is held out from a
clone of the winner, and the threshold whose validation recall lands
closest to 0.5 is stored in the model card as decision_threshold. In
the 2026-10-15 run that threshold is 0.13 (validation recall 0.497 on
15,734 rows). Measured once on the test patients, it gives recall
0.499, precision 0.193, and F1 0.278, flagging 5,746 of 19,802
encounters. The full threshold table (0.3, 0.5, tuned) is stored in
artifacts/metrics.json under threshold_analysis.

## Alternatives considered

- Keep 0.5. Rejected above: recall 0.027 makes the flag decorative.
- Retrain with class weights or resampling so 0.5 works. That changes
  the model and its probabilities to fix what is really a threshold
  problem. The ranking the model produces is fine; the cutoff was not.
- Pick the threshold that maximizes F1. Defensible, but it optimizes a
  blend when the stated job of the flag is screening. Targeting a
  recall level says plainly what the flag is for, and the precision
  cost is printed next to it in the model card.

## Consequences

- Roughly 29 percent of encounters get flagged, and about four of five
  flags are false alarms at current precision. Any consumer of this API
  has to staff for that volume; the README says so in plain terms.
- Retraining re-tunes the threshold. If the data shifts, the threshold
  moves with it, and the change shows up as a diff in metrics.json.
- The risk_band field (low, moderate, high) stays a display heuristic
  around the base rate. It is not the flag and is not used for any
  decision.
