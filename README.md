# Deep-Learning-Journey
The curated list of deep learning resources

# TOC

- [Deep Learning foundation](https://github.com/SueGK/Deep-Learning-Journey/edit/main/README.md#deep-learning-foundation)
- [Deep Learning Visualization](#visualization)
- [Pytorch](#pytorch)
  * [Tutorial](#tutorial)
  * [Code template & example](https://github.com/SueGK/Deep-Learning-Journey/edit/main/README.md#code-template--example)
- [Tensorflow](#tensorflow)
  * [Tutorial](#tutorial-1)
  * [Code template & example](https://github.com/SueGK/Deep-Learning-Journey/edit/main/README.md#code-template--example-1)
- [Paper](#paper)
- [Computer Vision](https://github.com/SueGK/Deep-Learning-Journey/edit/main/README.md#computer-vision)
- [Tool](#tool)
  * [Library](#library)
  * [Cheetsheet](#cheetsheet)
- [MLOps](#mlops)
  * [Docker](#docker)


# Deep Learning foundation

* [The spelled-out intro to neural networks and backpropagation: building micrograd - YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)
  * This is the most step-by-step spelled-out explanation of backpropagation and training of neural networks. It only assumes basic knowledge of Python and a vague recollection of calculus from high school.
  * Links:
     - micrograd on github: https://github.com/karpathy/micrograd
     - jupyter notebooks I built in this video: [randomfun/lectures/micrograd at master · karpathy/randomfun](https://github.com/karpathy/randomfun/tree/master/lectures/micrograd)
     - my website: https://karpathy.ai
     - my twitter: https://twitter.com/karpathy

# Visualization

* [Convolution Visualizer](https://ezyang.github.io/convolution-visualizer/index.html): 
  * This interactive visualization demonstrates how various convolution parameters affect shapes and data dependencies between the input, weight and output matrices.
  * ![CleanShot 2022-08-10 at 15 19 52](https://user-images.githubusercontent.com/71711489/183925490-abc8c619-4aab-4663-a654-facf045c28b1.gif)
* [julrog/nn_vis: A project for processing neural networks and rendering to gain insights on the architecture and parameters of a model through a decluttered representation.](https://github.com/julrog/nn_vis)
* [Convolution Neural Network Visualization - Made with Unity 3D and lots of Code](https://www.reddit.com/r/MachineLearning/comments/leq2kf/d_convolution_neural_network_visualization_made/) 
  * ![CleanShot 2022-08-10 at 15 00 56](https://user-images.githubusercontent.com/71711489/183921554-cb756522-f8ed-4fde-99a6-aa415aafc307.gif)
  * [Stefan Sietzen || Visuality](https://vimeo.com/stefsietz) 
* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
  * ![CleanShot 2022-08-10 at 15 23 11](https://user-images.githubusercontent.com/71711489/183926850-dca26c6b-781e-434d-9b91-c5dcfb5755e2.gif)


# Pytorch

## Tutorial

* [Deep Learning with PyTorch: A 60 Minute Blitz — PyTorch Tutorials 1.12.0+cu102 documentation](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html): Pytorch official tutorial

* [deeplizard - PyTorch - Python Deep Learning Neural Network API](https://www.youtube.com/watch?v=v5cngxo4mIg&list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG)
  * Deep explaination of tensor
  * <img src="https://user-images.githubusercontent.com/71711489/183897714-9e8b8508-2bc3-4bfd-b61e-600c4bd47711.png" width="400">
  * [My Course Notes](https://github.com/SueGK/Deep-Learning-Journey/blob/main/DeepLizard.md)

* [Practical Deep Learning for Coders - Practical Deep Learning](https://course.fast.ai/)
  * A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.
  * After finishing this course you will know:
    - How to train models that achieve state-of-the-art results in:
    - Computer vision, including image classification (e.g., classifying pet photos by breed)
    - Natural language processing (NLP), including document classification (e.g., movie review sentiment analysis) and phrase similarity
    - Tabular data with categorical data, continuous data, and mixed data
    - Collaborative filtering (e.g., movie recommendation)
    - How to turn your models into web applications, and deploy them
    - Why and how deep learning models work, and how to use that knowledge to improve the accuracy, speed, and reliability of your models
    - The latest deep learning techniques that really matter in practice
    - How to implement stochastic gradient descent and a complete training loop from scratch
    
  * [![FastAPI](https://img.shields.io/badge/FastAPI-0.63.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com) [![pytorch](https://img.shields.io/badge/PyTorch-1.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

* [datawhalechina/thorough-pytorch: PyTorch入门教程](https://github.com/datawhalechina/thorough-pytorch): Chinese pytorch tutorial

## Code template & example
* [jcjohnson/pytorch-examples: Simple examples to introduce PyTorch](https://github.com/jcjohnson/pytorch-examples)
 * <img src="https://user-images.githubusercontent.com/71711489/183907356-d696c286-3666-4718-a870-f70519eab1d9.png" width="400">

# Tensorflow

## Tutorial

## Code template & example

* [MrGemy95/Tensorflow-Project-Template: A best practice for tensorflow project template architecture.](https://github.com/Mrgemy95/Tensorflow-Project-Template#project-architecture): a tensorflow project template that combines simplcity, best practice for folder structure and good OOP design. 

# Paper

* [Browse the State-of-the-Art in Machine Learning | Papers With Code](https://paperswithcode.com/sota)
* [The latest in Machine Learning | Papers With Code](https://paperswithcode.com/)
* [Stateoftheart AI](https://www.stateoftheart.ai/): An open-data and free platform built by the research community to facilitate the collaborative development of AI
* [labmlai/annotated_deep_learning_paper_implementations: 🧑‍🏫 59 Implementations/tutorials of deep learning papers with side-by-side notes 📝](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

# Computer Vision
* [dmlc/gluon-cv: Gluon CV Toolkit](https://github.com/dmlc/gluon-cv): 
  * GluonCV provides implementations of the state-of-the-art (SOTA) deep learning models in computer vision.
  * ![short_demo](https://user-images.githubusercontent.com/71711489/183931315-ceb4c332-3a47-471b-8c5b-da64b18c2f7d.gif)

# Tool

## Library

[ml-tooling/best-of-ml-python: 🏆 A ranked list of awesome machine learning Python libraries. Updated weekly.](https://github.com/ml-tooling/best-of-ml-python)

## Cheetsheet

* [wzchen/probability_cheatsheet: A comprehensive 10-page probability cheatsheet that covers a semester's worth of introduction to probability.](https://github.com/wzchen/probability_cheatsheet)

* [The Ultimate Docker Cheat Sheet | dockerlabs](https://dockerlabs.collabnix.com/docker/cheatsheet/)

# MLOps

## MLOps Course
* [DataTalksClub/mlops-zoomcamp: Free MLOps course from DataTalks.Club](https://github.com/DataTalksClub/mlops-zoomcamp)
* [Course 2022 - Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)

## Docker

* [collabnix/dockerlabs: Docker - Beginners | Intermediate | Advanced](https://github.com/collabnix/dockerlabs)

* [docker/awesome-compose: Awesome Docker Compose samples](https://github.com/docker/awesome-compose)



