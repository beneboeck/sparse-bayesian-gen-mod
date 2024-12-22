# Sparse Bayesian Generative Modeling
Welcome to the repository for the paper "Sparse Bayesian Generative Modeling for Compressive Sensing"! The repository might be extended in the future.

Source code of the paper 
>B. Böck, S. Syed, and W. Utschick,
>"Sparse Bayesian Generative Modeling for Compressive Sensing," in Advances in
Neural Information Processing Systems, vol. 38, 2024.
<br>
Link to the paper: https://openreview.net/forum?id=GqefKjw1OR


## Abstract

This work addresses the fundamental linear inverse problem in compressive sensing (CS) by introducing a new type of regularizing generative prior. Our proposed method utilizes ideas from classical dictionary-based CS and, in particular, sparse Bayesian learning (SBL), to integrate a strong regularization towards sparse solutions. At the same time, by leveraging the notion of conditional Gaussianity, it also incorporates the adaptability from generative models to training data. However, unlike most state-of-the-art generative models, it is able to learn from a few compressed and noisy data samples and requires no optimization algorithm for solving the inverse problem. Additionally, similar to Dirichlet prior networks, our model parameterizes a conjugate prior enabling its application for uncertainty quantification. We support our approach theoretically through the concept of variational inference and validate it empirically using different types of compressible signals. 

## Requirements
The code is tested with `python 3.10`, `pytorch 2.5.1` and `pytorch-cuda 12.1`. Additional required packages comprise `pywavelets`, `scikit-image`, and `scikit-learn`.

## Instructions
The file `main_example_MNIST.ipynb` is a jupyter notebook which should make it easier to familiarise yourself with the code. The directory `modules` contains our proposed CSVAE and CSGMM implementations as well as the baselines SBL and LASSO. In `utils` you find some auxiliary functions for, e.g., generating the dictionary. In the directory `data` you can download or copy your data. 

## Citation
If you are using this code for your research, please cite

```bibtex

@inproceedings{boeck2024sparse,
	title={Sparse {B}ayesian Generative Modeling for Compressive Sensing},
	author={Benedikt B{\"o}ck and Sadaf Syed and Wolfgang Utschick},
	booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
	year={2024},
	url={https://openreview.net/forum?id=GqefKjw1OR}
}
