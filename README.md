# Fast-OSCAR-and-OWL-Regression-via-Safe-Screening-Rules
This repository contains the implementation for the ICML paper: Fast OSCAR and OWL Regression via Safe ScreeningRules. See our paper [http://proceedings.mlr.press/v119/bao20b.html](http://proceedings.mlr.press/v119/bao20b.html) for more details. 

# Introduction
The implementation in this repository solves OWL regression (aka SLOPE) as
$$
  \min\limits_{\beta}  P_{\lambda}(\beta):=\frac{1}{2}\|y-X\beta\|^2_{2} +\sum_{i=1}^d\lambda_{i}|\beta|_{[i]},
$$
where $X = [x_{1},x_{2},\ldots,x_{d}] \in \mathbb{R}^{n \times d}$ is the design matrix, $y \in \mathbb{R}^{d}$ is the measurement vector, $\beta$ is the unknown coefficient vector of the model, $\lambda = [\lambda_{1}, \lambda_{2}, \ldots, \lambda_{d}]$ is a non-negative regularization parameter vector of $d$ non-increasing weights and $|\beta|_{[1]} \geq |\beta|_{[2]} \geq \ldots \geq |\beta|_{[d]}$ are the ordered coefficients in absolute value. 

# Requirements
To use the solver, it is necessary to run makemex script in Matlab to  compile the Mex interface. To setup a compiler, type 'mex -setup' in the Matlab command line.


The script runSTR.m runs four functions to solve OWL regression, which includes 1) FISTA: 2) FISTA with Screen; 3) SVRG, 4) SVRG + screening. 