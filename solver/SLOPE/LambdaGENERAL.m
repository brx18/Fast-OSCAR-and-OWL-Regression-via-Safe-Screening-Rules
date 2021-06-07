function [lambda, kink] = LambdaGENERAL(n, q, Lgths, w, lam_type)

% Lambda sequence for equal groups sizes case. 
% n: number of rows
% q: target FDR level
% Lghts: vector of groups sizes
% w: vector of groups weights

if nargin ==4
    lam_type = 'gaussian';
end

%% Objects
eps              = 1e-8;
Lgths            = ( Lgths(:) )';
m                = length(Lgths);
w                = ( w(:) )';
lambda           = zeros(1,m);
critical_pvalues = (1:m)*q/m;

%% Finding the first lambda
y_targ1  = 1-critical_pvalues(1);
x_guess1 = 0;
y_guess1 = 0;
SCALE1   = w.^-1;
add = 0.5;

while y_guess1 <= y_targ1
    add = 2*add;
    x_guess1 = x_guess1 + add;
    y_guess1 = mean(chi2cdf( x_guess1^2./(SCALE1.^2), Lgths ));
end
a1 = x_guess1 - add;
b1 = x_guess1;

while abs(y_guess1 - y_targ1)>eps;
    c1=(a1+b1)/2;
    y_guess1 = mean(chi2cdf( c1^2./(SCALE1.^2), Lgths )); 
    if y_guess1 > y_targ1
        b1 = c1;
    else
        a1 = c1;
    end
end
lambda(1) = c1;

%% Finding rest lambdas
endd = 0;
ii   = 2;
while endd == 0;
    y_targ  = 1-critical_pvalues(ii);
    S       = ii-1;
    lam_cur = lambda(1:(ii-1));
    if strcmp(lam_type, 'orthogonal')
        SCALE = w.^-1;
    else
        SCALE   = w.^-1.*sqrt( (n-mean(Lgths)*S)/n+w.^2*(norm(lam_cur,2).^2)/(n-mean(Lgths)*S-1) );
    end
    x_guess = 0;
    y_guess = 0;
    add = 0.5;

    while y_guess <= y_targ
        add = 2*add;
        x_guess = x_guess + add;
        y_guess = mean(chi2cdf( x_guess^2./(SCALE.^2), Lgths ));
    end
    a = x_guess - add;
    b = x_guess;

    while abs(y_guess - y_targ)>eps;
        c=(a+b)/2;
        y_guess = mean(chi2cdf( c^2./(SCALE.^2), Lgths )); 
        if y_guess > y_targ
            b = c;
        else
            a = c;
        end
    end
    lambda(ii) = c;

    if lambda(ii) > lambda(ii-1) || ii>=m
        endd = 1;
    end
    ii = ii + 1;
end

lambda((ii-1):end) = lambda(ii-2);
kink = ii - 2;

end
