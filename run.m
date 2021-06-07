
clear all;

path_func0 = 'solver/SLOPE';
path_func1 = 'solver/STR';
addpath(path_func0);
addpath(path_func1);

n = 1000;
d = 10000;
A  = randn(n,d);
b  = randn(n,1);

gamma = 0.00001;
options.gamma = gamma;
options.batch = 50;
options.loop = n/options.batch;

lambda1_max = max(A'*b);
p = 0.2;
lambda1 = p*lambda1_max;
lambda2 = lambda1/(d);
for i = 1:d
    lambda(i) = lambda1 + (d-i)*lambda2;
end

[x1,info1] = APGD_STR(A,b,lambda,options);
[x2,info2] = APGDScreen_STR(A,b,lambda,options);
[x3,info3] = SVRG_STR(A,b,lambda,options);
[x4,info4] = SVRGScreen_STR(A,b,lambda,options);






