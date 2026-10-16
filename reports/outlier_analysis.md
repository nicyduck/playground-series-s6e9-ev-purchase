# Outlier analysis

## IQR rule

| Feature | Train range | IQR bounds | Flagged rows | Flagged rate | Test outside train range |
| --- | --- | --- | ---: | ---: | ---: |
| Age | 25–69 | 3–91 | 0 | 0.000% | 0 |
| Annual_Income_USD | 30,000–188,549 | 14,310.5–155,818.5 | 3,678 | 0.550% | 0 |
| Daily_Commute_km | 5.0–98.7 | -28.1–92.7 | 41 | 0.006% | 2 |
| Number_of_Cars_Owned | 1–4 | -0.5–3.5 | 13,489 | 2.017% | 0 |
| Charging_Stations_Near_Home | 0–14 | -5.5–14.5 | 0 | 0.000% | 0 |
| Charging_Stations_Near_Work | 0–19 | -7.5–20.5 | 0 | 0.000% | 0 |
| Environmental_Concern_Level | 1–5 | -1–7 | 0 | 0.000% | 0 |

`Number_of_Cars_Owned=4` is a valid discrete category, not a data error. It
should not be removed merely because the IQR heuristic flags it.

Rows flagged by any IQR rule account for 17,128 rows (2.562%). The current best
OOF model scores AUC `0.95066` on flagged rows versus `0.94464` on unflagged
rows. Their threshold-0.5 error rates are also similar (10.05% vs 9.87%). There
is no evidence that removing or clipping these rows would help.

## Rare exact values

For the lookup strategy, low-frequency exact values are more relevant than
large or small numeric values.

- Income values occurring fewer than 10 times cover 14,240 train rows.
- Only 1,655 test rows (0.578%) have an income value unseen in train.
- Only 33 test rows (0.012%) have a commute value unseen in train.
- Every test age occurs in train.

The smoothed lookup implementation already shrinks rare-value target rates
toward the global prior and falls back to the prior for unseen values. This is
preferable to deleting or winsorizing statistically extreme rows.
