# VectorRytov

Demo codes associated with the paper "Rytov Approximation of Vector Waves by Modifying Scattering Matrices: Precise Reconstruction of Dielectric Tensor Tomography". 

Please download all files in this repository before downloading the dataset from google drive
(https://drive.google.com/drive/folders/12nEj2LzxdJvx_EUq2GvkdWJwvtkllWV9?usp=drive_link)
and run
- main_SDS_MEMS.m to obtain the result shown in FIG. 2(b),
- main_PVA_MEMS.m to obtain the result shown in FIG. 2(c),
- main_LCN_DMD.m to obtain the result shown in FIG. 3.

The code for matrix unwrapping (unwrap_phase.m) has been adapted from 

M. A. Herraez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat, 
"Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path", 
Applied Optics, Vol. 41, Issue 35, pp. 7437-7444 (2002). 

M. F. Kasim's 'Fast 2D phase unwrapping implementation in MATLAB' (2017), which can be found at https://github.com/mfkasim91/unwrap_phase/.
