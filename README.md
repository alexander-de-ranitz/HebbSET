# HebbSET
This is the repository for my Bachelor Thesis titled 'Enhancing Learning in Sparse Neural Networks:  A Hebbian Learning Approach'. This repository contains my thesis describing my research and its results, as well as the Python implementation. For more information regarding the implementation and usage of the code, please see [my thesis](DeRanitz_Alexander_Thesis.pdf). For any questions or comments, please reach out to me via alexanderderanitz@gmail.com.

The code used in this repository is an extension of the SET implementation as found [here](https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks/tree/master/SET-MLP-Sparse-Python-Data-Structures). For further information regarding SET and its implementation, see:

Mocanu, D.C., Mocanu, E., Stone, P., Nguyen, P.H., Gibescu, M., Liotta, A.: _Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science_. Nature Communications 9 (2018).
https://doi.org/10.1038/s41467-018-04316-3

## Using the Code
The main functionality of HebbSET is found in [the set_mlp_sparse_data_structures.py file](set_mlp_sparse_data_structures.py). This file contains the main logic for creating, training, and evaluating the network.

Training data is contained in the Data folder. The [dataManager class](dataManager.py) is responsible for retrieving and pre-processing the data.

Lastly, two functions are written in the [Cython](https://github.com/alexander-de-ranitz/HebbSET/blob/main/sparseoperations.pyx). These functions are used to compute the backpropagation and Hebbian updates for the truly sparse neural network. Prior to use, this file needs to be compiled by running ```cythonize -a -i sparseoperations.pyx```. If you are unable to get Cython to work, equivalent functions are provided in the main python file. To use these, please comment out the Cython calls and uncomment the respective Python variants. However, please note that this will result in significantly longer running times.

In general, due to the truly sparse nature of the code, this implementation is significantly slower than what you might expect from state-of-the-art GPU implementations. When experimenting with these sparse neural networks, consider using small datasets (such as the provided Lung dataset) or using only part of a large dataset in order to ensure runtimes are reasonable.

