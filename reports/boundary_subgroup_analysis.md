# Boundary subgroup analysis

Scope: `Subsidy_Available=Yes` and `Environmental_Concern_Level in {3,4,5}`.

- Rows: 249,790
- Buyers: 111,660
- Non-buyers: 138,130
- Purchase rate: 44.70%

## Strongest remaining signals

| Feature | Buyer pattern | Non-buyer pattern | Strength |
| --- | --- | --- | --- |
| Annual_Income_USD | mean $98,763 | mean $79,888 | strongest numeric; univariate AUC 0.6855 |
| Range_Anxiety_Level | Low: 47.44% buy; Medium: 13.16%; High: 0.48% | anxiety strongly suppresses purchase | Cramér's V 0.1886 |
| Home_Charging_Possible | Yes: 48.08% buy | No: 35.94% buy | Cramér's V 0.1094 |
| Daily_Commute_km | mean 30.28 km | mean 32.64 km | weak; univariate AUC 0.5349 |
| City_Type | Rural 47.13%, Suburban 45.43%, Urban 42.89% | urban slightly less likely | weak; Cramér's V 0.0331 |
| Current_Car_Type | Truck 40.83% | other types around 44.3–45.5% | weak; Cramér's V 0.0211 |

Age, car count, and nearby charging-station counts have very small marginal
differences in this subgroup.

## Patterns after holding concern level fixed

Income quintile purchase rates:

| Income quintile | Concern 3 | Concern 4 | Concern 5 |
| --- | ---: | ---: | ---: |
| Q1 low | 6.10% | 15.94% | 38.56% |
| Q2 | 15.01% | 34.20% | 64.60% |
| Q3 | 18.81% | 40.38% | 70.24% |
| Q4 | 25.22% | 50.06% | 77.03% |
| Q5 high | 35.19% | 61.73% | 84.93% |

Home charging purchase rates:

| Home charging | Concern 3 | Concern 4 | Concern 5 |
| --- | ---: | ---: | ---: |
| No | 13.47% | 30.45% | 60.74% |
| Yes | 20.88% | 43.50% | 72.44% |

Range-anxiety purchase rates:

| Range anxiety | Concern 3 | Concern 4 | Concern 5 |
| --- | ---: | ---: | ---: |
| Low | 20.24% | 42.77% | 71.88% |
| Medium | 4.29% | 10.19% | 28.48% |
| High | 0.00% | 0.00% | 2.40% |

These gradients remain within every concern level, so they are not merely caused
by mixing concern levels 3, 4, and 5. The most promising explicit interaction
features are income × concern, range anxiety × concern, and home charging ×
concern. Income × range anxiety and income × home charging are also plausible
follow-up tests.
