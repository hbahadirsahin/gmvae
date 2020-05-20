# Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders

## Update 20-05-2020

- Accuracy metrics are added into code. It only uses accuracy metrics when a dataset has labels in it otherwise logs will print 0.
- I want to add "[normalized mutual information](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score)" metric to determine clustering performance, too. But I don't give promises since I am still same lazy person that haven't pushed any update to my NLP side project since months =) 
- Changes from informal README to semi-formal README format =) But I still hate all versions of tensorflow =)

## Introduction

This repository contains Tensorflow 2 version of [Deep Unsupervised Clustering with Gaussian Variatonal Autoencoders](https://arxiv.org/pdf/1611.02648.pdf) paper as an interview task.

I was asked to implement the idea/model/stuff in this paper and test it on a given dataset. This dataset contains 5M data points with only 1 feature and no labels. Hence, as my loss I use mean squared root error. For cross-check, I tested the implemented code on MNIST, but using MSE loss again instead of a categorical loss function. 

I do not guarantee this code can replicate the results in the paper, but loss function for both dataset decreased through iterations. Hence, I can say that this implementation works. 

## How-to-Run

You can either run this code via your favourite IDE or in console without any external parameters. Instead of parsing external  arguments/parameters, I created a configuration file in JSON format to load parameters . 

`./main.py` is the main code. To execute this project, you need to provide a valid `./config/config.json` file which contains the necessary configuration properties. 

In short,
- You can run this code pressing "Run" button on main.py in your IDE.
- You can run this code typing `python main.py` to your command window/terminal.

I tested this project in two different computers and have not encountered any path related problems. However, things may happen and you may need to do some extra path changes in code. But, I am pretty sure you have enough talent to read error messages and find which paths you need to change =)

## Referances
- https://github.com/RuiShu/vae-clustering
- https://github.com/jariasf/GMVAE/tree/master/tensorflow


