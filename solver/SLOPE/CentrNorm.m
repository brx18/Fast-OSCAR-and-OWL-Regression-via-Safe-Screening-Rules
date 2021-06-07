function X_CN = CentrNorm(X)

X_CN = bsxfun(@minus, X, mean(X));
X_CN = bsxfun( @rdivide, X_CN, sqrt(sum(bsxfun( @times, X_CN, X_CN))));

end

