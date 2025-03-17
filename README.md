# Sparse Bayesian Generative Modeling for Compressive Sensing
Welcome to the repository for the paper "Sparse Bayesian Generative Modeling for Compressive Sensing"! The repository might be extended in the future.

Source code of the paper 
>B. Böck, S. Syed, and W. Utschick,
>"Sparse Bayesian Generative Modeling for Compressive Sensing," in Advances in
Neural Information Processing Systems, vol. 38, 2024.
<br>
Link to the paper (Openreview): https://openreview.net/forum?id=GqefKjw1OR
Link to the paper (NeurIPS Proceedings): https://proceedings.neurips.cc/paper_files/paper/2024/hash/0857833a490eff6b49ce43eba1d01e8e-Abstract-Conference.html


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
 author = {B\"{o}ck, Benedikt and Syed, Sadaf and Utschick, Wolfgang},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {4629--4659},
 publisher = {Curran Associates, Inc.},
 title = {Sparse Bayesian Generative Modeling for Compressive Sensing},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/0857833a490eff6b49ce43eba1d01e8e-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}

```
## Licence of Contributions
This code is covered by the BSD 3-Clause License:

> BSD 3-Clause License
>
> Copyright (c) 2023 Benedikt Böck.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>  list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>  this list of conditions and the following disclaimer in the documentation
>  and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>  contributors may be used to endorse or promote products derived from
>  this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
