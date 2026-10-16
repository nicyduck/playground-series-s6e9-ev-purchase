# EDA summary

- train shape: `(668665, 15)`
- test shape: `(286571, 14)`
- target: `Will_Buy_EV`
- train missing cells: `0`
- test missing cells: `0`
- exact train duplicates (including id): `0`
- exact test duplicates (including id): `0`
- feature duplicates in train (excluding id/target): `0`
- feature duplicates in test (excluding id): `0`

## Target distribution

| Will_Buy_EV | count | ratio |
| --- | --- | --- |
| No | 551886 | 0.8253549983923191 |
| Yes | 116779 | 0.17464500160768098 |

## Column schema

| index | dtype | nunique | missing |
| --- | --- | --- | --- |
| id | int64 | 668665 | 0 |
| Age | int64 | 45 | 0 |
| Annual_Income_USD | float64 | 13214 | 0 |
| Daily_Commute_km | float64 | 805 | 0 |
| Number_of_Cars_Owned | int64 | 4 | 0 |
| Charging_Stations_Near_Home | int64 | 15 | 0 |
| Charging_Stations_Near_Work | int64 | 20 | 0 |
| Environmental_Concern_Level | float64 | 5 | 0 |
| Gender | str | 3 | 0 |
| City_Type | str | 3 | 0 |
| Current_Car_Type | str | 4 | 0 |
| Home_Charging_Possible | str | 2 | 0 |
| Subsidy_Available | str | 2 | 0 |
| Range_Anxiety_Level | str | 3 | 0 |
| Will_Buy_EV | str | 2 | 0 |

## Categorical train/test comparison

| column | train_nunique | test_nunique | test_only_levels | total_variation |
| --- | --- | --- | --- | --- |
| Gender | 3 | 3 | [] | 0.000199958607957928 |
| City_Type | 3 | 3 | [] | 0.0013468939989440265 |
| Current_Car_Type | 4 | 4 | [] | 0.0007498103523291798 |
| Home_Charging_Possible | 2 | 2 | [] | 0.0006618235805879169 |
| Subsidy_Available | 2 | 2 | [] | 0.0014545106686335574 |
| Range_Anxiety_Level | 3 | 3 | [] | 0.001710963145337423 |

## Numeric train/test comparison

| column | train_mean | test_mean | train_std | test_std | ks_stat | ks_pvalue |
| --- | --- | --- | --- | --- | --- | --- |
| Age | 47.03917059 | 47.07082364 | 12.87544825 | 12.86485034 | 0.00229309 | 0.2417706 |
| Annual_Income_USD | 84769.2669887 | 84799.50843247 | 28648.02904243 | 28626.85109753 | 0.00163921 | 0.65321743 |
| Daily_Commute_km | 32.1582975 | 32.15930502 | 18.73047388 | 18.72443589 | 0.00106705 | 0.97616521 |
| Number_of_Cars_Owned | 1.7126259 | 1.71609479 | 0.72927545 | 0.73108813 | 0.00166045 | 0.63717141 |
| Charging_Stations_Near_Home | 4.96040768 | 4.95069285 | 3.92684315 | 3.92862627 | 0.00157866 | 0.69883617 |
| Charging_Stations_Near_Work | 7.176314 | 7.15562635 | 5.18662665 | 5.18688039 | 0.00300461 | 0.05335512 |
| Environmental_Concern_Level | 2.93547741 | 2.93455374 | 1.42911874 | 1.42754125 | 0.00127138 | 0.9015695 |
