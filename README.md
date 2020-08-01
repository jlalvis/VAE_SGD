## Inversion with VAE and a new stochastic gradient descent.
This is the code associated to the manuscript **Deep generative models in inversion: a review and development of a new approach based on a variational autoencoder** which may be found here. An `environment.yml` file is provided and may be used as:

>conda env create -f environment.yml

to install the needed dependencies. Note that if you use pytorch with a GPU it sometimes works better to use the `-c pytorch` channel.

Once the dependencies are installed, you may install the code and run the synthetic tests in the manuscript available in each of the jupyter notebooks (.ipynb).

A brief explanation of contents:
- *SGD_DGM.py* : main module to run inversion with DGMs.
- *test_models* : models files for testing (as .npy)
- *toy_problems* : jupyter notebooks for illustrative 'toy' problems.
- *SGAN* : files needed to run comparison with SGAN and taken from here.
- *VAE* : files for training and generation of VAE in proposed approach.
