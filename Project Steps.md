## Project Steps & Results

#### Baseline - ResNet50 - SPIKaMed Whole Images					:white_check_mark: TOP1ACC on test = 85.15%

| predict\truth | Dys-  | Koi-  | Met-  | Par- | Sup- |
| :-----------: | :---: | :---: | :---: | :--: | :--: |
| Dyskeratotic  | 0.869 |  0.0  | 0.130 | 0.0  | 0.0  |
| Koilocytotic  | 0.04  | 0.64  | 0.24  | 0.04 | 0.04 |
|  Metaplastic  | 0.036 | 0.036 | 0.928 | 0.0  | 0.0  |
|   Parabasal   |  0.0  |  0.0  |  0.0  | 1.0  | 0.0  |
|  Superficial  |  0.0  |  0.0  |  0.0  | 0.0  | 1.0  |

#### Baseline - ResNet50 - SPIKaMed Cropped Images			  ​ :white_check_mark: TOP1ACC on test = 95.84%

| predict\truth | Dys-  | Koi-  | Met-  | Par- | Sup-  |
| :-----------: | :---: | :---: | :---: | :--: | :---: |
| Dyskeratotic  | 0.963 | 0.036 |  0.0  | 0.0  |  0.0  |
| Koilocytotic  | 0.024 | 0.915 | 0.036 | 0.0  | 0.024 |
|  Metaplastic  |  0.0  | 0.037 | 0.963 | 0.0  |  0.0  |
|   Parabasal   |  0.0  |  0.0  |  0.0  | 1.0  |  0.0  |
|  Superficial  | 0.012 |  0.0  |  0.0  | 0.0  | 0.988 |

#### Baseline2 -  Finetune Baseline w cropped SPIKaMed model on Smear dataset

(Trained and test only, train:test = 1:9)





#### Model 1 - ResNet50 - Masked Region of SPIKaMeD Whole Images

####  								  																							:white_check_mark: TOP1ACC on test = 87.82%

| predict\truth | Dys-  | Koi-  | Met-  | Par-  | Sup- |
| :-----------: | :---: | :---: | :---: | :---: | :--: |
| Dyskeratotic  | 0.869 | 0.087 | 0.043 |  0.0  | 0.0  |
| Koilocytotic  | 0.08  | 0.64  | 0.12  |  0.0  | 0.16 |
|  Metaplastic  | 0.035 | 0.107 | 0.821 | 0.035 | 0.0  |
|   Parabasal   | 0.167 |  0.0  |  0.0  | 0.833 | 0.0  |
|  Superficial  |  0.0  |  0.0  |  0.0  |  0.0  | 1.0  |



#### Model 2 - 

- Train the whole image on ResNet50 with Attention layer
- Test on the whole image

#### Graph CNN


