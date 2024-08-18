##Code and README file are for the moment messay. On Wednesday 21st of August, this should be fixed.

---

This repository was written in the scope of the Master thesis "Adversarial Robustness in Generative - Restricted Kernel Machines".
It contains tools to generate adversarial examples against deep generative models and assess the performance of the models. These tools are combined into one class called "class_Adv_attack_genRKM".
Moreover, it comprises a latent analysis of the genRKM trying to discover the origin of the adversarial robustness.

Authors: 

Naichuan Zhang and Grégoire Corlùy, students in Master of Statistics and Data Science at the KU Leuven.

Purpose Master thesis:

In this thesis, the goal is to improve and analyze the adversarial robustness of the genRKM. 
The analysis can be divided in four distinct parts from which the first three are included in this repository:

1) Compare the performance of a vanilla VAE with a vanilla genRKM against state-of-the-art attacks to observe which model has inherently adversarial robustness properties
2) Compare the performance of an adversarially-trained VAE with an adversarially-trained genRKM to see which model enjoys the most from this effective adversarial defence
3) Perform a latent space 

Models:

The genRKM is based on the RKM framework

Explanation content:

every folder what you can do with it



Credits: 

The code builds up on the code from Arun Pandey concerning the genRKM. For the VAE, the code is inspired from the one of Jackson Kang.

Link to repository of Pandey: https://www.esat.kuleuven.be/stadius/E/pandey/softwareGenRKM.php

Link to repository of Jackson Kang: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial


References:

cite in APA style
[1] Pandey, A., Schreurs, J., & Suykens, J. A. (2021). Generative restricted kernel machines: A framework for multi-view generation and disentangled feature learning. Neural Networks, 135, 177-191.

To add to code:
The algorithm was taken from a \href{https://github.com/MadryLab/mnist_challenge}{Github repository} that implemented the adversarial training of Madry on the MNIST dataset. The cross-entropy loss comes from the \texttt{torch.nn} library with the function $CrossEntropyLoss()$.\\ for the adversarial training


