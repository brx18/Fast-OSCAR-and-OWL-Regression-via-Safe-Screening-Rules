function [STRUCT, r_idxs, pvalues] = Struct(X, y, rho, CORR)

%This function divides set of whole considered variables into groups.
%Hierarchy comes from p-values given by simple linear regression tests
%which were applicated on which variable independendly.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS:
%X - data mamatrix, each column has to be normalized, i.e. has mean equal
%to zero and l_2 norm equal to 1;
%y - vector of observations.
%rho - lower limit of correlation, gives one from criteria of group
%creating;
%CORR - matrix of correlations.
%OUTPUTS:
%STRUCT - structure of data
%r_idxs - vector of indices of representatives in whole group of columns
%pvalues - p-values for representatives (in corresponding order)

%% Objects
[n,p]        = size(X);
STRUCT       = zeros(1,p);
r_idxs       = zeros(1,p);
pvalues      = zeros(1,p);
gr_nmb       = 1;
if rho==1
    rho=1-1e-10;
end

% Simple linear regression tests
beta_LS = (y-mean(y))'*X;
RSS     = bsxfun( @minus, y, bsxfun(@times, X, beta_LS));
RSS     = sum(bsxfun( @times, RSS, RSS));
SE      = sqrt(RSS/(n-2));
t       = beta_LS./SE;
pv      = 2-2*tcdf(abs(t),n-2);

[spv, comp]   = sort(pv);

while ~isempty(comp)
    idx1            = comp(1);
    r_idxs(gr_nmb)  = idx1;
    pvalues(gr_nmb) = abs(spv(1));
    group_ind       = find(abs(CORR(idx1,comp))>=rho);
    group           = comp(group_ind);
    comp(group_ind) = [];
    spv(group_ind)  = [];
    STRUCT(group)   = gr_nmb;  
    gr_nmb          = gr_nmb + 1;

end

r_idxs  = r_idxs(r_idxs>0);
pvalues = pvalues(pvalues>0);

end