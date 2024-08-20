## Code and README file are for the moment messy. On Wednesday 21st of August 2024, this should be fixed.

---

This repository was written in the scope of the Master thesis "Adversarial Robustness in Generative Restricted Kernel Machines" under the supervision of Prof. Suykens.
It contains tools to generate adversarial examples against deep generative models and assess the performance of the models. These tools are combined into one class called "class_Adv_attack_genRKM".
Moreover, it comprises a latent analysis of the genRKM trying to discover the origin of the adversarial robustness.

Table of contents:

Link to the different chapters

Authors: 

Naichuan Zhang and Grégoire Corlùy, students in Master of Statistics and Data Science at the KU Leuven.

Promotor: Prof. dr. ir. Johan Suykens
Daily advisors: ir. Sonny Achten and ir. Bram De Cooman

Purpose Master thesis:

In this thesis, the goal is to improve and analyze the adversarial robustness of the genRKM, ideally using the good properties of this particular model. 
The analysis can be divided in four distinct parts from which the first three are included in this repository (corresponding folder given between parenthesis):

1) Compare the performance of a vanilla VAE with a vanilla genRKM against state-of-the-art attacks to observe which model has inherently adversarial robustness properties (Comparison-vanilla-models)
2) Compare the performance of an adversarially-trained VAE with an adversarially-trained genRKM to see which model enjoys the most from this effective adversarial defence (Comparison-adv-trianed-models)
3) Perform a latent space analysis by comparing the latent space of a vanilla and adversarially-trained genRKM, trying to understand the origin of the advserial robustness (Latent-space-analysis)
4) Comparison of the newly implemented robust genRKM models with the vanilla and adversarially-trained VAE and genRKM (previously introduced) (See https://github.com/zncQueiros/Adversarial_Robustness_Generatieve_RKM.git)

Models:

The genRKM [1] is based on the RKM framework [2], invented by Johan Suykens, which uses the synergies of the kernel PCA, Least-Squares Support Vector Machines and Restricted Boltzmann Machine.
It is a generative model having as interesting properties the latent space disentanglement and the multiview.
The architecture of the genRKM is given below.

![Architecture of the genRKM with the bottom nodes being the input, the intermediate nodes the features being computed by a neural network and the top nodes corresponding to the latent space by combining the different views.](images/genRKM multiview-Pandey.jpg)

Tools:

The adversarial attacks and some adversarial defenses implemented in this thesis are based on state-of-the-art attacks and defenses.
PGD attack for adversarial training: Madry [3]
Attack in the latent space: Tabacof [4] (implemented, but not used in the analysis)
Untargeted and targeted type 2 attacks: Sun [5]

Metrics: 

Adversarial attacks: To measure the similarity between images, three metrics were used: Frobenius norm, Structural Similarity Index Measure (SSIM) and Learned Perceptual Image Patch Similarity (LPIPS).
Latent space analysis: To measure the similarity between latent vectors, the cosine similarity and the eucidean distance were used as metrics.

Explanation content:

every folder what you can do with it



Credits: 

The code builds up on the code from Arun Pandey concerning the genRKM. For the VAE, the code is inspired from the one of Jackson Kang.

Link to repository of Pandey: https://www.esat.kuleuven.be/stadius/E/pandey/softwareGenRKM.php

Link to repository of Jackson Kang: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial


References:

cite in APA style
[1] Pandey, A., Schreurs, J., & Suykens, J. A. (2021). Generative restricted kernel machines: A framework for multi-view generation and disentangled feature learning. Neural Networks, 135, 177-191.
[2] Suykens, J. A. (2017). Deep restricted kernel machines using conjugate feature duality. Neural computation, 29(8), 2123-2163.
[3] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083.
[4] Tabacof, P., Tavares, J., & Valle, E. (2016). Adversarial images for variational autoencoders. arXiv preprint arXiv:1612.00155.
[5] Sun, C., Chen, S., Cai, J., & Huang, X. (2020, October). Type I attack for generative models. In 2020 IEEE international conference on image processing (ICIP) (pp. 593-597). IEEE.

To add to code:
The algorithm was taken from a \href{https://github.com/MadryLab/mnist_challenge}{Github repository} that implemented the adversarial training of Madry on the MNIST dataset. The cross-entropy loss comes from the \texttt{torch.nn} library with the function $CrossEntropyLoss()$.\\ for the adversarial training

Contact: Gregoire.stephane.corluy@ulb.be
