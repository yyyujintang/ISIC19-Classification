# ISIC19-Classification

This is a project for semi-supervised medical image classification.

ISIC-19 Data Website: https://challenge2019.isic-archive.com/

I divide the data into train:valid:test as 7:2:1, and choose 10%/20%/40% of the train data as labeld data, the others train data as unlabeld data.

Data Preparation:

[01_Data_Partrition.ipynb](https://github.com/yyyujintang/ISIC19-Classification/blob/main/01_Data_Partrition.ipynb)

[03_10_20_30_partrition.ipynb](https://github.com/yyyujintang/ISIC19-Classification/blob/main/03_10_20_30_partrition.ipynb)

Network(ipynb format):

[02_ResNet50.ipynb](https://github.com/yyyujintang/ISIC19-Classification/blob/main/02_ResNet50.ipynb)

Train:

[train_newest.py](https://github.com/yyyujintang/ISIC19-Classification/blob/main/train_newest.py)

Test

[test.py](https://github.com/yyyujintang/ISIC19-Classification/blob/main/test.py)

This is only the basic code for the program.

Some results is also included.
