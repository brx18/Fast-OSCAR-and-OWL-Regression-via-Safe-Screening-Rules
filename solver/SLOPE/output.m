function X=output(A,a)

%This function adds create results matrix

[w,c] = size(A);
if isempty(a)
    a=NaN;
end
la    = length(a);
if la<w
    a=[a;NaN(w-la,1)];
    A=[A,a];
else
    A=[A;NaN(la-w,c)];
    A=[A,a];
end
X=A;
end