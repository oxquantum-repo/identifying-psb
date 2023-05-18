# Identifying Pauli spin blockade using deep learning
Code for publication 'Identifying Pauli spin blockade using deep learning' (https://arxiv.org/abs/2202.00574)

The full data set is published in https://doi.org/10.5281/zenodo.7948852


# Installation

In an environment of you choice, run ```pip install -r requirements.txt``` to install the required packages for this publication. 

If you want to read raw data, there is a caveat: Most of the raw data is taken with Igor and requires the ```igor``` module. Unfortunately, it is not being actively maintained and has a compatiblity issue with newer numpy versions. If you are using numpy>=1.20 (as required by other modules and stated in the requirements) you need to manually replace ```_numpy.complex``` with just ```complex``` in the ```binarywave.py``` code of the ```igor``` module. Please get in touch if you have troubles with this.


# Content
This is all code you need to recreate the results from the publication mentioned above.

## Data
This folder contains raw and processed data for the training of the neural networks and its evaluation, as well as information about training results and trained neural network weights. The full data set is published in https://doi.org/10.5281/zenodo.7948852. Folders with data in them have a jupyter notebook that illustrates how to access the data.


## Augmentation
For training the neural network with experimental data, we will need to augment the data to create a larger data set. The code for this is here, together with a jupyter notebook that demonstrates it.


## Simulator
Simulated data can be created with the code provided here. There is a jupyter notebook that demos the simulator including an interactive part.

## Training
The training of the neural networks is demonstrated here, including data handling.

## Visualisation
How to show results from the training and other plots are in this folder.
