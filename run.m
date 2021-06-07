
clear all;

path_func0 = 'solver/utils';
path_func1 = 'solver/method';
addpath(path_func0);
addpath(path_func1);

n = 1000;
d = 10000;
A  = randn(n,d);
b  = randn(n,1);

lambda1_max = max(A'*b);
p = 0.2;
lambda1 = p*lambda1_max;
lambda2 = lambda1/(d);
for i = 1:d
    lambda(i) = lambda1 + (d-i)*lambda2;
end

[x1,info1] = APGD(A,b,lambda);
[x2,info2] = APGDScreen(A,b,lambda);
[x3,info3] = SPGD(A,b,lambda);
[x4,info4] = SPGDScreen(A,b,lambda);






