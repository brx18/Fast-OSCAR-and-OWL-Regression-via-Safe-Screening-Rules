function beta_supp = BETA_SUPP(p, k, space)

start = randsample(space,1);
gr_n = floor((p-start)/space);
gridd = (0:(gr_n))*space+start;

if length(gridd)<k
    error('k too large or space too large');
end
beta_supp = sort(randsample(gridd,k));

end