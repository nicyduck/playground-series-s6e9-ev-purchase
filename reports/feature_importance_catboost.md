# CatBoost feature importance

5-foldベースラインの `PredictionValuesChange` 重要度をfold間で平均した順位。
重要度はモデル内で予測値を変えるために各特徴が使われた度合いであり、因果効果ではない。

| Rank | Feature | Mean importance | Fold std |
| ---: | --- | ---: | ---: |
| 1 | Subsidy_Available | 52.325981 | 1.145580 |
| 2 | Environmental_Concern_Level | 18.571476 | 0.220581 |
| 3 | Annual_Income_USD | 9.132193 | 0.087279 |
| 4 | Age | 3.845625 | 0.179476 |
| 5 | Daily_Commute_km | 3.788964 | 0.144453 |
| 6 | Range_Anxiety_Level | 3.288958 | 0.250431 |
| 7 | Charging_Stations_Near_Home | 2.399376 | 0.470723 |
| 8 | Charging_Stations_Near_Work | 2.324159 | 0.364332 |
| 9 | Current_Car_Type | 1.150749 | 0.248369 |
| 10 | Number_of_Cars_Owned | 1.114623 | 0.244174 |
| 11 | Home_Charging_Possible | 1.008757 | 0.125673 |
| 12 | City_Type | 0.539272 | 0.127210 |
| 13 | Gender | 0.509867 | 0.055583 |
