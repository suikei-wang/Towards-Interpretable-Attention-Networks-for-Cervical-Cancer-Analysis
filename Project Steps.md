## Project Steps

#### Baseline - ResNet50 - Whole Images			#Done, TOP1ACC = 85%#

#### Baseline2 - ResNet - Segmentated Patches

- Train on masked images and only learn useful part of the image (Mask CNN w ResNet50?)
- Test on masked whole images

#### Generalization test

- Finetune the pre-trained model in Baseline2 on Smear2005 (try 7:3 train and test firstly, then apply 5:95 train and test to see the result)

#### Better CNN

- Train the whole image on ResNet50 with Attention layer
- Test on the whole image

#### Graph CNN

