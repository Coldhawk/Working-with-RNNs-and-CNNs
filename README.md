# Working with GANs and PINNs
This project was created for the course Deep Learning for Science (YMX8170).

The project involves two tasks: 
1. Using an Autoencoder for Data Analysis.
2. Using a Physics Informed Neaural Network to solve an equation.

Task 1: Autoencoders and GAN
This task involves using an AutoEncoder, Variational Autoencoder and a GAN for the Y. Isaienkov LEGO Minigfigures pictures dataset. 
The dataset is divided into train, test and validation loaders and train images are augmented to generate more input data.

The Autoencoder takes 224p images and transforms them into a (56 * 28 * 28,4) size latent space. The loss platues after around 30 epochs and generates very blurry images. The latent space seems to only include images from two franchises, Marvel and Star Wars. Earlier versions included all four (Marvel, Star Wars, Harry Potter, Jurassic World).

The Variational Autoencoder also takes 224p images and transforms them into (16, 256 * 7 * 7) linear space. The reconstruction loss and KL divergence platues after 20 epochs and the model also generates blurry images. The latent space also only has Marvel and Star Wars, but the targets seem to be correlated.

The GAN is a basic adversarial model that uses spherical coordinates, otherwise is standard. For the GAN Optuna hyperparameter tuning was used but this did not really change the results. The loss for the discriminator rises f
