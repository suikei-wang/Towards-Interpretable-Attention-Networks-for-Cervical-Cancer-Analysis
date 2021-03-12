## Project Steps

#### Baseline - ResNet50 - Whole Images			#Done, TOP1ACC = 85%#

#### Baseline2 - ResNet - Segmentated Patches

- Train a segmentation model on the Smear dataset (with masked label)
- Apply the segmentation model to the SPIKaMeD dataset, now we also got the masked label
- Train ResNet50 on the masked region of whole image only
- Test the model on masked  whole images

#### Generalization test

- Finetune the pre-trained model in Baseline2 on Smear2005 for **classification** (try 7:3 train and test firstly, then apply 5:95 train and test to see the result)

#### Better CNN

- Train the whole image on ResNet50 with Attention layer
- Test on the whole image

#### Graph CNN

