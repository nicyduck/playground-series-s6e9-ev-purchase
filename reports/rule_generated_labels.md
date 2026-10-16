# Rule-generated labels

- OOF ROC AUC: `0.93671021`
- Accuracy at 0.5: `0.89592397`
- Generated Yes rate: `0.14896996`
- TN / FP / FN / TP: `525674 / 26212 / 43380 / 73399`
- Fold AUCs: `[0.9351653108946936, 0.9365509316267686, 0.9376747485760875, 0.9375625414762191, 0.9367754230191683]`

## Feature importance

```text
remainder__Environmental_Concern_Level    5.025502e-01
cat__Subsidy_Available_Yes                2.567348e-01
remainder__Annual_Income_USD              1.089676e-01
cat__Subsidy_Available_No                 9.997337e-02
cat__Range_Anxiety_Level_Low              3.176939e-02
cat__Home_Charging_Possible_No            4.062624e-06
cat__Home_Charging_Possible_Yes           5.080572e-07
cat__Range_Anxiety_Level_High             0.000000e+00
cat__Range_Anxiety_Level_Medium           0.000000e+00
```

## Final-tree rules

```text
|--- remainder__Environmental_Concern_Level <= 4.50
|   |--- remainder__Environmental_Concern_Level <= 3.50
|   |   |--- remainder__Environmental_Concern_Level <= 2.50
|   |   |   |--- cat__Subsidy_Available_Yes <= 0.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 135317.00
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 75628.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  75628.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  135317.00
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 144992.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  144992.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |--- cat__Subsidy_Available_Yes >  0.50
|   |   |   |   |--- remainder__Environmental_Concern_Level <= 1.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 102591.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  102591.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Environmental_Concern_Level >  1.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 107503.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  107503.50
|   |   |   |   |   |   |--- class: 0
|   |   |--- remainder__Environmental_Concern_Level >  2.50
|   |   |   |--- cat__Subsidy_Available_Yes <= 0.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 91350.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 66505.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  66505.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  91350.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 138723.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  138723.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |--- cat__Subsidy_Available_Yes >  0.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 100002.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 67506.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  67506.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  100002.50
|   |   |   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |   |   |--- class: 0
|   |--- remainder__Environmental_Concern_Level >  3.50
|   |   |--- cat__Subsidy_Available_No <= 0.50
|   |   |   |--- remainder__Annual_Income_USD <= 77624.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 67517.00
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 50994.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  50994.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  67517.00
|   |   |   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |--- remainder__Annual_Income_USD >  77624.50
|   |   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 108978.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  108978.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 107715.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  107715.00
|   |   |   |   |   |   |--- class: 1
|   |   |--- cat__Subsidy_Available_No >  0.50
|   |   |   |--- remainder__Annual_Income_USD <= 101926.00
|   |   |   |   |--- remainder__Annual_Income_USD <= 66911.00
|   |   |   |   |   |--- cat__Home_Charging_Possible_Yes <= 0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- cat__Home_Charging_Possible_Yes >  0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  66911.00
|   |   |   |   |   |--- cat__Home_Charging_Possible_No <= 0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- cat__Home_Charging_Possible_No >  0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |--- remainder__Annual_Income_USD >  101926.00
|   |   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 119327.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  119327.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 126952.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  126952.50
|   |   |   |   |   |   |--- class: 0
|--- remainder__Environmental_Concern_Level >  4.50
|   |--- cat__Subsidy_Available_Yes <= 0.50
|   |   |--- remainder__Annual_Income_USD <= 107390.00
|   |   |   |--- remainder__Annual_Income_USD <= 67712.50
|   |   |   |   |--- cat__Home_Charging_Possible_No <= 0.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 51348.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  51348.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- cat__Home_Charging_Possible_No >  0.50
|   |   |   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |--- remainder__Annual_Income_USD >  67712.50
|   |   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 81087.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  81087.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 69067.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  69067.00
|   |   |   |   |   |   |--- class: 0
|   |   |--- remainder__Annual_Income_USD >  107390.00
|   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |--- class: 0
|   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 124014.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 115927.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  115927.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  124014.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 131164.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  131164.50
|   |   |   |   |   |   |--- class: 0
|   |--- cat__Subsidy_Available_Yes >  0.50
|   |   |--- remainder__Annual_Income_USD <= 67506.50
|   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 51857.00
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  51857.00
|   |   |   |   |   |--- class: 0
|   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 50943.00
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 46457.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  46457.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  50943.00
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 62001.00
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- remainder__Annual_Income_USD >  62001.00
|   |   |   |   |   |   |--- class: 0
|   |   |--- remainder__Annual_Income_USD >  67506.50
|   |   |   |--- cat__Range_Anxiety_Level_Low <= 0.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 109575.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 84558.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  84558.50
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- remainder__Annual_Income_USD >  109575.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 122279.00
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- remainder__Annual_Income_USD >  122279.00
|   |   |   |   |   |   |--- class: 1
|   |   |   |--- cat__Range_Anxiety_Level_Low >  0.50
|   |   |   |   |--- remainder__Annual_Income_USD <= 100002.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 84950.00
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- remainder__Annual_Income_USD >  84950.00
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- remainder__Annual_Income_USD >  100002.50
|   |   |   |   |   |--- remainder__Annual_Income_USD <= 126965.00
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- remainder__Annual_Income_USD >  126965.00
|   |   |   |   |   |   |--- class: 1
```
