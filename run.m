% This file is a demo of safe screening for OWL regression for our ICML paper: Fast OSCAR and OWL Regression via Safe Screening Rules at http://proceedings.mlr.press/v119/bao20b.html
% APGD: An accelerated proximal gradient method (FISTA-type) in \cite{bogdan2015slope}
% APGDScreen: APGD method with our proposed screening;
% SPGD: An proximal stochastic variance-reduced method (ProxSVRG) in \cite{xiao2014proximal}.;
% SPGDScreen: An ProxSVRG method with our proposed screening.

clear all;

path_func0 = 'solver/utils';
path_func1 = 'solver/method';
addpath(path_func0);
addpath(path_func1);

% generate the data
n = 1000;
d = 10000;
A  = randn(n,d);
b  = randn(n,1);

% generate the regularization parameters
lambda1_max = max(A'*b);
p = 0.2;
lambda1 = p*lambda1_max;
lambda2 = lambda1/(d);
for i = 1:d
    lambda(i) = lambda1 + (d-i)*lambda2;
end

% methods
[x1,info1] = APGD(A,b,lambda);
[x2,info2] = APGDScreen(A,b,lambda);
[x3,info3] = SPGD(A,b,lambda);
[x4,info4] = SPGDScreen(A,b,lambda);






