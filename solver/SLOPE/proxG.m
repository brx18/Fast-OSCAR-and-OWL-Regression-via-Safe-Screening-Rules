function x = proxG(y, lambda, I, I2)

% This function calculate prox for mean-group SLOPE. The fast prox
% algorithm from SLOPE Toolbox version 1.0. is used.
%
% I provides information about structure of data.  It should be vector such
% as indices given by find(I==i) correspond to ith group.
% L is vector of sizes of groups

%% Objects
m      = size(I2,2);
y      = y(:);
lambda = lambda(:);

%% First step
y2  = [(y.^2)',0];
y_I = (sum(y2(I2),1))';
y_I = sqrt(y_I);

% Normalization
[y_I_s,idx] = sort(abs(y_I),'descend');

% Compute solution and re-normalize
c = zeros(m,1);
v = proxSortedL1Mex(y_I_s,lambda);
c(idx) = v;

%% Second step
multiplier = c./y_I;
multiplier = multiplier(I);

x = y.*multiplier;

end
