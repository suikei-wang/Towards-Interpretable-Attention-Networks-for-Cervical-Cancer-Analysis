## Project Steps

#### Baseline - ResNet50 - Whole Images													:white_check_mark: TOP1ACC = 85%#

#### Baseline2 - ResNet - Segmentated Patches

- Get the masked label of SPIKaMeD 												   	:white_check_mark: 
- Train ResNet50 on the masked region of whole image only
- Test the model on masked  whole images

#### Generalization test

- Finetune the pre-trained model in Baseline2 on Smear2005 for **classification** (try 7:3 train and test firstly, then apply 5:95 train and test to see the result)

#### Better CNN

- Train the whole image on ResNet50 with Attention layer
- Test on the whole image

#### Graph CNN

