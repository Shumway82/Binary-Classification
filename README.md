## Binary Classifier

# Deevio Data Scientist Hiring Challenge

## Introduction
Modern manufacturing processes have a high degree of automation and at the same time high
quality requirements. Machine vision helps ensuring these quality requirements are met by
providing means for automatic recognition of defects. However, traditional approaches for
machine vision require experts to implement the necessary algorithms, take time to set up, and
need fine tuned lighting installations.
Deevio sets out to simplify the machine vision process drastically to bring automated quality
control to broad application areas within a short time-to-market.
We have successfully applied our machine vision technology to multiple use cases from different
domains. Currently, we are building a prototype consisting of a camera and a computing device
which can be deployed in a factory setting at a pilot customer. The goal of Deevio is to provide
technology that covers different use cases including classification, defect localization and
counting. In the future, more data sources from the manufacturing process can be tapped to
correlate visible occurrence with other production parameters. Thus, Deevio will be able to give
detailed real-time insights into the production process.

## The Challenge
The challenge is an opportunity to demonstrate proficiency with problem solving and
collaboration skills we would expect you to use as Data Scientist at Deevio. It consists of two
tasks and is meant as the foundation for further onsite collaboration during your interview.
Additionally, we want you to get a feel for some of the problems and tasks you'll encounter at
Deevio.

## Questions to keep in mind
#### How well does your model perform? How does it compare to the simplest baseline model you can think of?
####  How many images are required to build an accurate model?
####  Where do you see the main challenge in building a model like the one we asked here?
####  What would you do if you had more time to improve the model?
####  What problems might occur if this solution would be deployed to a factory that requires automatic nails quality assurance?

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

## Pre-requiremtents
* tensorflow >= 1.8 
* pil 
* numpy 
* opencv

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
$ pip3 install tfcore
or for editing the repository 
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
$ python pipeline_inferencer.py --dataset "../Data/" --model_dir "Models_Deevio_Nailgun" 
```
