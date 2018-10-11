# Data Scientist - Challenge

## Introduction
Modern manufacturing processes have a high degree of automation and at the same time high
quality requirements. Machine vision helps ensuring these quality requirements are met by
providing means for automatic recognition of defects. However, traditional approaches for
machine vision require experts to implement the necessary algorithms, take time to set up, and
need fine tuned lighting installations.
The Company sets out to simplify the machine vision process drastically to bring automated quality
control to broad application areas within a short time-to-market.
We have successfully applied our machine vision technology to multiple use cases from different
domains. Currently, we are building a prototype consisting of a camera and a computing device
which can be deployed in a factory setting at a pilot customer. The goal of the company is to provide
technology that covers different use cases including classification, defect localization and
counting. In the future, more data sources from the manufacturing process can be tapped to
correlate visible occurrence with other production parameters. Thus, the company will be able to give
detailed real-time insights into the production process.

## The Challenge
The challenge is an opportunity to demonstrate proficiency with problem solving and
collaboration skills we would expect you to use as Data Scientist at the Company. It consists of two
tasks and is meant as the foundation for further onsite collaboration during your interview.
Additionally, we want you to get a feel for some of the problems and tasks you'll encounter at
the company.

## Questions to keep in mind
#### How well does your model perform? How does it compare to the simplest baseline model you can think of?
My model have on the training-set nealy 100% accuracy and on the test-set an accurays of 77%. The model tends to overfitting. To prevent this I use dropouts and weight decay. To avoid overfitting I choose an really simple model with very less parameter and an small input image size. Experiments with more parameter and bigger images increase the overfitting problem. The reason for overfitting in this case is to less training-data.
####  How many images are required to build an accurate model?
To create an accurate model more than 5k images are needed. To overcome this issue I use data augmentation. The most meaningful technique in this case is flipping(horizontal and vertical) and rotation with a step-angle of 1 and reflection. Transformations are not usefull.  
####  Where do you see the main challenge in building a model like the one we asked here?
Tuning hyperparameter. To overcome the overfitting problem it is importend to use the right parameter count in the model architecture,  and to find the right regularization factors. An cyclic learning-rate is usefull to overcome local minima (high learning rate) and to find the best optimum (low learning rate). 
####  What would you do if you had more time to improve the model?
To find the best model I would study and use "Neural Architecture Search" methods like AutoML (Google). An other methode is using GAN's to create synthetic samples if more data are needed. GAN's can also used as unsupervised classifier. In this case one generator for creating synthetic data and two discriminator, one discriminator for classification and one discriminator for training the generator. Transfer-Learning can be used by using the first layers of existing models to overcome the problem with limited data. I would implement an better and more robust preprocessing pipeline.
####  What problems might occur if this solution would be deployed to a factory that requires automatic nails quality assurance?
The factory usually has very high speeds in the production pipeline. The processing must be very fast which makes it necessary to process both the preprocessing and the classification above 50-100Hz. Other problems are different light situations or motion bluring if you have high processing- and to low capturing frequencys. The localisation of the nails are really importend for the classification task afterwards. A robust preprocessing pipeline is needed.

## Preprocessing
1. Central cropping of input-image to image size 960x960
2. Find nail by treshold segmentation. \
   2.1 Final treshold T = max(image) - 32 \
   2.2 False if image(x,y) < T else True 
3. Find central mass point center_x and center_y of True-Area
4. Central cropping at center_x and center_y with size 448x448
5. Downscalling to 112x112 to decrease image size and to reduce noise of background

##### Cropped image to image size 448x448
![bad](https://github.com/Shumway82/Binary-Classification/blob/master/Data/images/image_bad_448.jpeg)

##### Final sample of good and bad nails 
![bad](https://github.com/Shumway82/Binary-Classification/blob/master/Data/images/image_bad_112.jpeg)
![bad](https://github.com/Shumway82/Binary-Classification/blob/master/Data/images/image_good_112.jpeg)

## Model-Architecture
1. conv5x5, filter=16, stride=1, padding=VALID, instance norm, relu
2. max_pool2x2 
3. dropout=0.5
4. conv5x5, filter=16, stride=1, padding=VALID, instance norm, relu
5. max_pool2x2 
6. dropout=0.5
7. conv5x5, filter=32, stride=1, padding=VALID, instance norm, relu
8. max_pool2x2 
9. dropout=0.5
10. conv3x3, filter=64, stride=1, padding=VALID, instance norm, relu
11. max_pool2x2
12. dropout=0.5
13. flatten
14. fc512, relu
15. fc512, relu
16. fc2
17. softmax

## Hyper-Parameter
Batch-size 16, instance norm for conv-layer, cyclic learning-rate 1e-4 - 1e-5, adam-optimizer with beta1 0.9, epochs 2000, training set = 160 images (50% good, 50% bad), test set = 38 images (50% good, 50% bad)

#### Accuracy of Test-Set and Training-Set
![bad](https://github.com/Shumway82/Binary-Classification/blob/master/Data/images/accuracy.png)

## Run Docker-Image
1. Clone the repository
```
$ git clone https://github.com/Shumway82/Binary-Classification.git
```
2. Go to folder
```
$ cd Binary-Classification
```
3. Build and run Docker
```
$ docker-compose build
$ docker-compose up
```
4. Run
```
$ http://localhost:5000/predict?image_url=https://raw.githubusercontent.com/Shumway82/Binary-Classification/master/Data/test/1522142328_bad.jpeg
```

## Installation tf_base package
1. Clone the repository
```
$ git clone https://github.com/Shumway82/tf_base.git
```
2. Go to folder
```
$ cd tf_base
```
3. Install with pip3
``` 
$ pip3 install -e .
```

## Install Binary-Classification package

1. Clone the repository
```
$ git clone https://github.com/Shumway82/Binary-Classification.git
```
2. Go to folder
```
$ cd Binary-Classification
```
3. Install with pip3
```
$ pip3 install -e .
```

## Usage-Example

1. Training
```
$ python pipeline_trainer.py --dataset "../Data/"
```

2. Inferencing
```
$ python pipeline_inferencer.py --dataset "../Data/" --model_dir "Nails" 
```
