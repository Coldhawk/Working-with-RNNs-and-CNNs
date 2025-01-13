# Working with GANs and PINNs
This project was created for the course Deep Learning for Science (YMX8170).

Credit:
- The Autoencoders and GAN are based on the work of Sebastian Raschka, the helper modules are also modified versions of his. [Link to source](https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L16/conv-autoencoder_mnist.ipynb)
- The Dataset is the Y. Isaienkov LEGO Minigfigures pictures dataset. [Link to source](https://www.kaggle.com/datasets/ihelon/lego-minifigures-classification/data)
- The PINN is based on the work of Ben Moseley. [Link to source](https://github.com/benmoseley/DLSC-2023/blob/main/lecture-5/PINN%20demo.ipynb)

The project involves two tasks: 
1. Using an Autoencoder for Data Analysis.
2. Using a Physics Informed Neaural Network to solve an equation.

*Note for Task 1: The jupyter notebook file is an older  version, html is up to date.* 

**Task 1: Autoencoders and GAN**

This task involves using an AutoEncoder, Variational Autoencoder and a GAN for the Y. Isaienkov LEGO Minigfigures pictures dataset. 
The dataset is divided into train, test and validation loaders and train images are augmented to generate more input data.

The Autoencoder takes 224p images and transforms them into a (56 * 28 * 28,4) size latent space. The loss platues after around 30 epochs and generates very blurry images. The latent space seems to only include images from two franchises, Marvel and Star Wars. Earlier versions included all four (Marvel, Star Wars, Harry Potter, Jurassic World).

The Variational Autoencoder also takes 224p images and transforms them into (16, 256 * 7 * 7) linear space. The reconstruction loss and KL divergence platues after 20 epochs and the model also generates blurry images. The latent space also only has Marvel and Star Wars, but the targets seem to be correlated.

The GAN is a basic adversarial model that uses spherical coordinates, otherwise is standard. For the GAN Optuna hyperparameter tuning was used but this did not really change the results. The loss for both the discriminator and generator rise fast but platue around epoch 110. Generator loss is around 1, Discriminator loss aroung 0.5. The generated images at epoch 150 are very blurry and noisy as with the autoencoders, but you can somewhat make out the shape of a minifigure.

**Task 2: PINN**
This task involves the use of a Neural Network to solve a physics problem, in this case the heat equation of a uniform bar of length $L$, where one end is place on the origin and the other at $x = L$.
The change in temperature is a partial differential equation which is a homogeneous inital boundary value problem solved with the Fourier method.

First the Fourier Series is approximated and $u(x,t)$ is show on a graph, where $t$ is shown in different colors. The equation reaches zero fast, by the time $t = 0.5$. Then the Fourier series is plotted for different numbers of terms. By the 5th term, the series is accurate to the solution so that is what is used.

A PINN is trained to simulate the system by setting up the exact solution, PINN architecture, loss function, boundary conditions and training data. After 15000 training steps, the model is mostly accurate, but seems to use a different term value for the Fourier series, as the curve doesn't line up 100% with the exact solution.

Afterwords, a PINN is trained to unvert for underlying parameters, in this case $a$ and $L$, the *material coefficient* and the *bar length*. Observational data is generated based on the exact solution and noise is added. This observational data doesn't follow the exact solution for some reason. 

The PINN is then trained on this observational data and the loss function inclued a MSE and the physics error. The results are not great, as the generated data is wrong.

