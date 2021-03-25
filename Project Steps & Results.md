## Project Steps & Results

#### Baseline1 - ResNet50 - SPIKaMed Full Images				

#### :white_check_mark: TOP1ACC on test = 85.15%

|              | Precision | Recall | F1-Score | Support |
| :----------: | :-------: | :----: | :------: | :-----: |
| Dyskeratotic |   0.869   | 0.869  |  0.869   |   23    |
| Koilocytotic |   0.889   |  0.64  |  0.744   |   25    |
| Metaplastic  |   0.764   | 0.928  |  0.839   |   28    |
|  Parabasal   |   0.923   |   1    |   0.96   |   12    |
| Superficial  |   0.846   | 0.846  |  0.846   |   13    |

| predict\truth | Dys-  | Koi-  | Met-  | Par- | Sup- |
| :-----------: | :---: | :---: | :---: | :--: | :--: |
| Dyskeratotic  | 0.869 |  0.0  | 0.130 | 0.0  | 0.0  |
| Koilocytotic  | 0.04  | 0.64  | 0.24  | 0.04 | 0.04 |
|  Metaplastic  | 0.036 | 0.036 | 0.928 | 0.0  | 0.0  |
|   Parabasal   |  0.0  |  0.0  |  0.0  | 1.0  | 0.0  |
|  Superficial  |  0.0  |  0.0  |  0.0  | 0.0  | 1.0  |

#### Baseline2 - ResNet50 - SPIKaMed Cropped Images			  

#### :white_check_mark: TOP1ACC on test = 95.84%

|              | Precision | Recall | F1-Score | Support |
| :----------: | :-------: | :----: | :------: | :-----: |
| Dyskeratotic |   0.975   | 0.951  |  0.963   |   82    |
| Koilocytotic |   0.895   | 0.928  |  0.911   |   83    |
| Metaplastic  |   0.95    |  0.95  |   0.95   |   80    |
|  Parabasal   |     1     | 0.988  |  0.994   |   80    |
| Superficial  |   0.988   | 0.988  |  0.988   |   84    |

| predict\truth | Dys-  | Koi-  | Met-  | Par- | Sup-  |
| :-----------: | :---: | :---: | :---: | :--: | :---: |
| Dyskeratotic  | 0.963 | 0.036 |  0.0  | 0.0  |  0.0  |
| Koilocytotic  | 0.024 | 0.915 | 0.036 | 0.0  | 0.024 |
|  Metaplastic  |  0.0  | 0.037 | 0.963 | 0.0  |  0.0  |
|   Parabasal   |  0.0  |  0.0  |  0.0  | 1.0  |  0.0  |
|  Superficial  | 0.012 |  0.0  |  0.0  | 0.0  | 0.988 |

#### Baseline3 -  Baseline w cropped SPIKaMed model on Smear dataset

(Train:test=1:9, 50 epochs training)

####  																																:white_check_mark: TOP1ACC on test =57.61%

|                     | Precision | Recall | F1-Score | Support |
| :-----------------: | :-------: | :----: | :------: | :-----: |
|      Carcinoma      |   0.532   |  0.7   |  0.604   |   120   |
|  Light Dysplastic   |   0.680   | 0.685  |  0.683   |   146   |
| Moderate Dysplastic |   0.419   | 0.402  |  0.410   |   117   |
|   Normal Columnar   |   0.521   | 0.481  |   0.5    |   79    |
| Normal Intermediate |   0.742   | 0.875  |  0.803   |   56    |
| Normal Superficiel  |   0.959   | 0.783  |  0.862   |   60    |
|  Severe Dysplastic  |   0.405   | 0.335  |  0.367   |   158   |

|    predict\truth    | Car-  | L-Dys- | M-Dys- | N-Col- | N-Inter- | N-Sup- | S-Dys |
| :-----------------: | :---: | :----: | :----: | :----: | :------: | :----: | :---: |
|      Carcinoma      | 0.683 | 0.016  | 0.041  | 0.025  |   0.0    |  0.0   |  0.0  |
|  Light Dysplastic   | 0.034 | 0.630  | 0.144  | 0.021  |  0.027   |  0.0   | 0.144 |
| Moderate Dysplastic | 0.094 | 0.256  | 0.427  | 0.008  |  0.017   |  0.0   | 0.197 |
|   Normal Columnar   | 0.278 | 0.051  | 0.025  | 0.494  |   0.0    |  0.0   | 0.152 |
| Normal Intermediate | 0.018 | 0.036  | 0.036  | 0.036  |  0.875   | 0.036  |  0.0  |
| Normal Superficiel  |  0.0  |  0.0   |  0.0   |  0.0   |  0.217   | 0.783  |  0.0  |
|  Severe Dysplastic  | 0.196 | 0.057  | 0.234  | 0.171  |   0.0    |  0.0   | 0.341 |



#### Baseline4 - ResNet50 - Masked Region of SPIKaMeD Full Images

####  								  																							:white_check_mark: TOP1ACC on test = 86.13%

|              | Precision | Recall | F1-Score | Support |
| :----------: | :-------: | :----: | :------: | :-----: |
| Dyskeratotic |   0.815   | 0.957  |   0.88   |   23    |
| Koilocytotic |   0.789   |  0.6   |  0.682   |   25    |
| Metaplastic  |   0.815   | 0.786  |   0.8    |   28    |
|  Parabasal   |   0.769   | 0.833  |   0.8    |   12    |
| Superficial  |    0.8    | 0.923  |  0.857   |   13    |

| predict\truth | Dys-  | Koi-  | Met-  | Par-  | Sup- |
| :-----------: | :---: | :---: | :---: | :---: | :--: |
| Dyskeratotic  | 0.957 | 0.043 |  0.0  |  0.0  | 0.0  |
| Koilocytotic  | 0.08  | 0.64  | 0.12  | 0.04  | 0.12 |
|  Metaplastic  | 0.035 | 0.107 | 0.821 | 0.035 | 0.0  |
|   Parabasal   | 0.167 |  0.0  |  0.0  | 0.833 | 0.0  |
|  Superficial  |  0.0  |  0.0  |  0.0  |  0.0  | 1.0  |



#### Model 1 - Residual Attention Network - SPIKaMeD Full Images

####  :white_check_mark: TOP1ACC on test = %

|              | Precision | Recall | F1-Score | Support |
| :----------: | :-------: | :----: | :------: | :-----: |
| Dyskeratotic |           |        |          |         |
| Koilocytotic |           |        |          |         |
| Metaplastic  |           |        |          |         |
|  Parabasal   |           |        |          |         |
| Superficial  |           |        |          |         |

| predict\truth | Dys- | Koi- | Met- | Par- | Sup- |
| :-----------: | :--: | :--: | :--: | :--: | :--: |
| Dyskeratotic  |      |      |      |      |      |
| Koilocytotic  |      |      |      |      |      |
|  Metaplastic  |      |      |      |      |      |
|   Parabasal   |      |      |      |      |      |
|  Superficial  |      |      |      |      |      |



#### Model 2 - DenseNet - SPIKaMeD Full Images

####  :white_check_mark: TOP1ACC on test = 89.11%

|              | Precision | Recall | F1-Score | Support |
| :----------: | :-------: | :----: | :------: | :-----: |
| Dyskeratotic |   0.846   | 0.957  |  0.898   |   23    |
| Koilocytotic |   0.864   |  0.76  |  0.809   |   25    |
| Metaplastic  |   0.929   | 0.929  |  0.929   |   28    |
|  Parabasal   |     1     | 0.917  |  0.957   |   12    |
| Superficial  |   0.857   | 0.923  |  0.889   |   13    |

| predict\truth | Dys-  | Koi-  | Met-  | Par-  | Sup-  |
| :-----------: | :---: | :---: | :---: | :---: | :---: |
| Dyskeratotic  | 0.957 | 0.043 |  0.0  |  0.0  |  0.0  |
| Koilocytotic  | 0.08  | 0.76  | 0.08  |  0.0  | 0.08  |
|  Metaplastic  | 0.035 | 0.035 | 0.929 |  0.0  |  0.0  |
|   Parabasal   | 0.083 |  0.0  |  0.0  | 0.917 |  0.0  |
|  Superficial  |  0.0  | 0.077 |  0.0  |  0.0  | 0.923 |







#### Graph CNN


