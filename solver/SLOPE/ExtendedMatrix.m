function XZ = ExtendedMatrix(X)

%% Objects
[n,p] = size(X);
Z     = zeros(n,p);

%% Loop
for ii=1:p
    a          = X(:,ii);
    b          = zeros(n,1);
    NZidxs     = find(a~=0);
    a_r        = a(NZidxs);
    b_r        = b(NZidxs);    
    a_ru       = unique(a_r);
    Oidxs      = a_r == a_ru(2);
    b_r(Oidxs) = 1;
    b_r        = b_r-mean(b_r);
    b_r        = b_r/norm(b_r);    
    b(NZidxs)  = b_r;
    Z(:, ii)   = b;
end

%% Summarize
XZ = [X,Z];

end
