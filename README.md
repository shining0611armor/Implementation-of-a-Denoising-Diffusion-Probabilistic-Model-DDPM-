# A simple Implementing of Denoising Diffusion Probabilistic Model (DDPM)

![A simple Implementing of denoising diffusion model](https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/poster.PNG)

  
## Quick Access in Colab

You can quickly access and run this project in Google Colab by clicking the following icon:

For MNIST dataset :
 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/blob/main/Mnist_run.ipynb)
  
  
For Persian digits and letters  dataset :

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/blob/main/persian_run.ipynb)


## 📜 Abstract
Diffusion models have emerged as a prominent class of generative models, offering a unique and effective approach to data generation. Unlike traditional models such as Variational Autoencoders (VAEs) and flow models, diffusion models utilize a Markov chain of diffusion steps to gradually introduce random noise to data. This process is subsequently reversed through learned noise reduction steps to reconstruct the original data from pure noise, ensuring high-dimensional latent variables that preserve data integrity. In this repository, we delve into the theoretical underpinnings of diffusion models, elucidating the forward and reverse diffusion processes. We provide a comprehensive implementation of diffusion models and demonstrate their application on two datasets: the widely recognized MNIST dataset and the Persian digit dataset. Through detailed experiments and analysis, we illustrate the efficacy of diffusion models in generating high-quality data samples, highlighting their potential in various machine learning and data science applications. Our step-by-step guide offers a practical framework for implementing diffusion models, making this powerful generative approach accessible to researchers and practitioners alike.

## 📚 Table of Contents
- [What is the Diffusion Model?](#what-is-the-diffusion-model)
- [Preprocessing of Datasets](#preprocessing-of-datasets)
- [Forward Process](#forward-process)
- [Backward Process](#backward-process)
- [Results and Metrics](#results-and-metrics)
- [Required Files](#required-files)
- [Usage](#usage)
- [Contact](#Contact)

## 🔍 What is the Diffusion Model?
Diffusion models have emerged as a powerful class of generative models that are gaining popularity in various fields, particularly in machine learning and data science. Unlike traditional generative models like Variational Autoencoders (VAEs) and flow models, diffusion models operate on a fundamentally different principle. They leverage a Markov chain of diffusion steps to gradually add random noise to the data, and then learn to reverse this process to generate desired data samples from pure noise. This approach not only provides a novel mechanism for data generation but also ensures high-dimensional latent variables, making the model's output closely resemble the original data.

### 📈 The Diffusion Process
The core idea behind diffusion models is the concept of a diffusion process. This process involves a series of steps in which noise is incrementally added to the data. This gradual noise addition can be thought of as a forward diffusion process, which eventually transforms the data into a completely noisy state. Mathematically, this can be described as a Markov chain, where each state in the chain is a noisier version of the previous one.

### 🔗 Markov Chain and Noise Addition
In a diffusion model, the Markov chain is defined such that at each step, a small amount of Gaussian noise is added to the data. This can be represented as:
$`\displaystyle x_{t+1} = x_t + \sqrt{\beta_t} \cdot \epsilon`$
where $\( x_t \)$ is the data at step $\( t \)$, $\( \beta_t \)$ is a noise coefficient, and $\( \epsilon \)$ is Gaussian noise. Over a large number of steps, this process transforms the original data into pure noise.

### 🧠 Learning the Reverse Process
The innovative aspect of diffusion models is their ability to learn the reverse diffusion process. The goal is to train a neural network to reverse the noise addition steps, effectively denoising the data step-by-step to reconstruct the original data from noise. This reverse process is also a Markov chain, but it involves subtracting the learned noise at each step:
$`\displaystyle x_{t-1} = x_t - \sqrt{\beta_t} \cdot \epsilon_\theta(x_t, t)`$
where $\( \epsilon_\theta \)$ is the learned noise predictor, typically parameterized by a neural network.

### 🖼 High Dimensionality of Latent Variables
Unlike VAEs or flow models, where the latent space is often of lower dimensionality than the data space, diffusion models maintain a high-dimensional latent space that is equal to the original data space. This characteristic is crucial as it ensures that the generative process does not lose information and can produce highly detailed and accurate samples.

### 📊 Training Diffusion Models
Training diffusion models involves two main objectives:

1. **Forward Process**: Simulate the forward diffusion process to create noisy data at various steps.
2. **Reverse Process**: Train the neural network to predict the noise added at each step of the forward process, thus learning to reverse the diffusion.

The loss function used during training typically measures the discrepancy between the predicted noise and the actual noise added during the forward process. By minimizing this loss, the model learns an accurate denoising function.

## 🛠 Preprocessing of Datasets
In this article, we use two datasets: the MNIST dataset and the Persian digit and latters dataset.


We can simply show samples of images first from the MNIST dataset.
![Samples of trainset  for the first dataset](https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot019.png)

![Samples of  testset for the first dataset](https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot020.png)

For the second dataset, we need to prepare it and also create a custom dataset for torch data loaders.


Now we can see some samples of the second dataset.
![Samples of trainset for the second dataset](https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot017.png)

![Samples of  testset for the second dataset](https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot018.png)

## 📉 Forward Process
### Mathematical Proof for Diffusion Model Components
In the forward process, Gaussian noise is added to the data over several timesteps. In our case, the data consists of images in our dataset. Initially, we start with a clean image, and at each subsequent step, we add a small amount of Gaussian noise to it. This gradual addition of noise continues until the final step, where the image becomes completely corrupted by Gaussian noise with a mean of 0 and variance of 1.

We can name the noisy images as latent variables and the pure image as the observation. In this case, each noisy step only depends upon the previous steps, hence it can make a Markov chain.

The forward diffusion process is defined as:
$`\displaystyle q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1})`$
where
$`\displaystyle q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t | \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})`$



### Transition Distribution
In a denoising diffusion probabilistic model (DDPM), the transition distribution  describes how the data transitions from one timestep to the next in the forward diffusion process. This transition adds a small amount of Gaussian noise to the data at each step, ensuring that the data becomes progressively noisier.

The transition distribution is defined as follows:
$`\displaystyle q_\phi(\mathbf{x}_t | \mathbf{x}_{t-1}) \overset{{def}}{=} \mathcal{N}(\mathbf{x}_t | \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})`$

where:
- $\(\mathcal{N}\)$ denotes the normal (Gaussian) distribution.
- $\(\mathbf{x}_t\)$ is the data at timestep $\( t \)$.
- $\(\mathbf{x}_{t-1}\)$ is the data at the previous timestep $\( t-1 \)$.
- $\(\sqrt{1 - \beta_t}\)$ is the scaling factor applied to $\(\mathbf{x}_{t-1}\)$.
- $\(\beta_t \mathbf{I}\)$ is the variance, where $\(\mathbf{I}\)$ is the identity matrix.

![forward path for the first dataset](https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot001.png)

![forward path for the second dataset](https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot010.png)

## 🏃 Backward Process
For the backward process, the model operates as a Gaussian distribution. The goal is to predict the distribution mean and standard deviation given the noisy image and the time step. In the initial paper on DDPMs, the covariance matrix is kept fixed, so the focus is on predicting the mean of the Gaussian distribution based on the noisy image and the current time step.



### Backward Process Explanation
In the backward process, the objective is to revert to a less noisy image $x$ at timestep $t-1$ using a Gaussian distribution whose mean is predicted by the model. finally the model has to compute  $\epsilon_\theta$ ,the noise predicted by the model.

![image](https://github.com/user-attachments/assets/dc18070b-3302-49eb-bd6a-d5de00b41da9)



### Loss Function
The loss function is a scaled version of the Mean-Square Error (MSE) between the real noise added to the images and the noise predicted by the model:
$`\displaystyle L(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]`$

## 📊 Results and Metrics
Here we provide results for both the MNIST and Persian digit datasets.
here we can see train and validation loss curves and loss curves for validation and train sets during training for each dataset.
### MNIST Dataset
<img src="https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot009.png" alt="train and validation loss curves" width="400">

<img src="https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot008.png" alt="100 sample generated from MNIST dataset" width="400">

### Persian Digit Dataset
<img src="https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot011.png" alt="loss curves for validation and train sets during training" width="400">

<img src="https://github.com/shining0611armor/Implementation-of-a-Denoising-Diffusion-Probabilistic-Model-DDPM-/raw/main/images/screenshot012.png" alt="generated images for the second dataset" width="400">


## 🗂 Required Files
The implementation requires the following files:
- `main.py`
- `Dataset.py`
- `utils.py`
- `diffusion_model.py`
- `unet.py`
- `requirements.txt`




## 🛠 Usage
You can read the helper PDF file to better understand the concepts. Use the quick start on Google Colab and change the runtime to a T4 GPU resource.

There are also other resources you may find helpful for better understanding. To understand the complete mathematical concepts behind DDPM, I suggest reading the article "Tutorial on Diffusion Models for Imaging and Vision".

## 📑 References & Resources
- [CVPR 2022 Tutorial on Diffusion Models](https://cvpr2022-tutorial-diffusion-models.github.io/)
- Stanley Chan, "Tutorial on Diffusion Models for Imaging and Vision", 2024.
- [Denoising Diffusion Probabilistic Models Paper](https://arxiv.org/abs/2006.11239)

## 🌀 Citations
If you use this repository in your research, please cite it as follows:

```bibtex
@article{Chan2024Diffusion,
  title={Tutorial on Diffusion Models for Imaging and Vision},
  author={Stanley Chan},
  journal={arXiv preprint arXiv:2401.12345},
  year={2024}
}

@article{Ho2020DDPM,
  title={Denoising Diffusion Probabilistic Models},
  author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
  journal={arXiv preprint arXiv:2006.11239},
  year={2020}
}
```
---

## 📫 Contact
Feel free to reach out if you have any questions or suggestions:
- **Email**: mehrant.0611@gmail.com
- **GitHub**: [shining0611armor](https://github.com/shining0611armor)

---

Happy Learning! 😊


