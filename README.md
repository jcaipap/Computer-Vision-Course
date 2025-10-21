#Computer Vision
This repository contains various codes and explanatory material from various practical assignments carried out in a course on computer vision. They explore topics in deep learning, particularly the use of convolutional neural networks, pose estimation, and iterative algorithms. The course is part of the third-year curriculum for general engineering students in the PGE program, specializing in sciences and techniques for numerical transformation, a specialization with a business component and a data analysis component. 

##Declaration
All content was created with the help of the professor ABABSA Fakhreddine, who provided the basic codes from previous complex projects and the theoretical explanations necessary to implement different Deep Learning functions. However, the codes were written from scratch, based on various standard functions and well-known CNNs. 

##Contents
###TP1: Convolutional neural networks (CNN). 
In this practical assignment, we will examine two different strategies for solving this problem:
• Training a new model from scratch with increased data,
• Feature extraction with a pre-trained network
As an example, we will focus on classifying images into dog images and cat images, using a training dataset containing 4,000 images of cats and dogs (2,000 cats, 2,000 dogs). Typical exercise of CNN Introduction.

###TP2: 3D Following with KLT-PNP
The purpose of this exercise is to study the processing chain required to perform 3D tracking of objects in a sequence of images in order to implement an
augmented reality application. In particular, we will see how to combine KLT with the Ransac-Pnp algorithm for 3D tracking, in a sequence of images, of a mobile camera relative to a straight pavement.

###TP3: CNN to 3D Points regression
In this tutorial, we propose to use a CNN to regress the coordinates of 3D points from a set of 2D RGB patches. The images are acquired by a mobile camera that moves freely around the working environment. 
The CNN model takes RGB image patches measuring 50 × 50 pixels as input. The convolutional layers learn the features present in these patches using filtering and pooling operations. The resulting vector of learned features is flattened and then used to adjust a multilayer perceptron (MLP). The latter is specifically designed to perform regression in order to obtain three-dimensional outputs corresponding to 3D coordinates in the world coordinate system. 

###TP4: ICP (Iterative Closest Point)
The goal of this practical assignment is to implement a method for registering 3D data as seen in class, in this case the ICP algorithm and its variants. In particular, we will see how to solve this popular estimation problem using Python on the one hand, and the Open3D library on the other. 
