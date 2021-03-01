## Inversion with VAE and a new stochastic gradient descent.
This is the code associated to the manuscript **Deep generative models in inversion: a review and development of a new approach based on a variational autoencoder** which may be found [here](https://arxiv.org/abs/2008.12056). An `environment.yml` file is provided and may be used as:

>conda env create -f environment.yml

to install the needed dependencies. Note that if you use pytorch with a GPU it sometimes works better to use the `-c pytorch` channel.

Once the dependencies are installed, you may install the code and run the synthetic tests in the manuscript available in each of the jupyter notebooks (.ipynb). The notebooks `DGM_inv_linear_comparison.ipynb` and `DGM_inv_nonlinear_comparison.ipynb` run the inversions once the DGMs are trained (they read the trained VAE from a parameter file with extension '.pth').

A brief explanation of contents:
- *SGD_DGM.py* : main module to run inversion with DGMs. This module defines and then takes inputs from a 'SGDsetup' object.
- *test_models* : models files for testing (as .npy).
- *toy_problems* : jupyter notebooks for illustrative 'toy' problems. The notebook `toy_SGD_deform.ipynb` contains the example for testing the performance of SGD with 'ring' regularization considering a misfit function with three local minima. The notebook `toy_VAE_eight.ipynb` contains the example of an 'eight-shaped' manifold being approximated with a VAE using different values of alpha and beta.
- *SGAN* : files needed to run comparison with SGAN and taken from [here](https://github.com/elaloy/gan_for_gradient_based_inv). In this way, the SGAN is already trained and its parameters come from the '.pth' file.
- *VAE* : files for training and generation of VAE in proposed approach. The jupyter notebook `training_VAE.ipynb` is used to train the VAE. A GPU is best for lower computational time in this part. The saved parameters for the VAE trained with alpha=0.1 and beta=1000 are provided for testing inversion directly.
