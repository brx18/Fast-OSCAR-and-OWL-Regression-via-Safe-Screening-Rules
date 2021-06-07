# Fast-OSCAR-and-OWL-Regression-via-Safe-Screening-Rules
This repository contains the implementation of safe screening for solving OWL regression (aka SLOPE) for the ICML paper: Fast OSCAR and OWL Regression via Safe Screening Rules. See our paper [http://proceedings.mlr.press/v119/bao20b.html](http://proceedings.mlr.press/v119/bao20b.html) for more details. 

# Requirements
To use the solver, please run makemex script in Matlab to compile the Mex interface.

# Structure of the repository
The script run.m runs four functions to solve OWL regression, which includes 1) APGD: 2) APGDScreen; 3) SPGD, 4) SPGDScreen. 

* APGD: An accelerated proximal gradient method (FISTA-type);
* APGDScreen: APGD method with our proposed screening;
* SPGD: An proximal stochastic variance-reduced method (ProxSVRG);
* SPGDScreen: An ProxSVRG method with our proposed screening.

Please find the reference in our paper.