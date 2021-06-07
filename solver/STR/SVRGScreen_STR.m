function [x,info] = SVRGScreen_STR(A,b,lambda,options)
% -------------------------------------------------------------
% Start timer
% -------------------------------------------------------------
t0 = cputime;

% -------------------------------------------------------------
% Parse parameters
% -------------------------------------------------------------
if (nargin <  4), options = struct(); end

iterations = getDefaultField(options,'iterations',100000);
verbosity  = getDefaultField(options,'verbosity',1);
fid        = getDefaultField(options,'fid',1);
optimIter  = getDefaultField(options,'optimIter',1);
gradIter   = getDefaultField(options,'gradIter',20);
screenIter   = getDefaultField(options,'screenIter',1);
tolInfeas  = getDefaultField(options,'tolInfeas',1e-9);
tolRelGap  = getDefaultField(options,'tolRelGap',1e-9);
xInit      = getDefaultField(options,'xInit',[]);


% -------------------------------------------------------------
% Initialize
% -------------------------------------------------------------

% Get problem dimension
d = size(A,2);
n = size(A,1);

% Get initial lower bound on the Lipschitz constant
s = RandStream('mt19937ar','Seed',0);
x = randn(s,d,1); x = x / norm(x,2);
x = A'*(A*x);
L = norm(x,2)*2;

% Constants for exit status
STATUS_RUNNING    = 0;
STATUS_OPTIMAL    = 1;
STATUS_ITERATIONS = 2;
STATUS_MSG = {'Optimal','Iteration limit reached'};

% Initialize parameters and iterates
if (isempty(xInit)), xInit = zeros(d,1); end

lambda  = lambda(:);
b       = b(:);
x       = xInit;
Ax      = A*x;
iter    = 0;
status  = STATUS_RUNNING;
Aprods  = 2;
ATprods = 1;
l2      = vecnorm(A,2)';
prev    = d;
curr    = d;
active  = (1:d)';


proxFunction = @(v1,v2) proxSortedL1(v1,v2);


if (verbosity > 0)
    fprintf(fid,'%5s  %9s   %9s  %9s  %9s\n','Iter','||r||_2','Gap','Infeas.','Rel. gap');
end

iGradF = @(Z, x, y, i) (Z(i,:))'*(Z(i,:)*(x-y));


gamma = options.gamma;
batch = options.batch;
P     = options.loop;


% -------------------------------------------------------------
% Main loop
% -------------------------------------------------------------
while (true)
    % Compute the gradient at f(y)
    r = Ax-b;
    g = A'*r;
    f = (r'*r) / 2;
    
    % Increment iteration count
    iter = iter + 1;
    
    % Check optimality conditions
    if ((mod(iter,optimIter) == 0))
        % Compute 'dual', check infeasibility and gap
        gs  = sort(abs(g),'descend');
        ys  = sort(abs(x),'descend');
        infeas = max(max(cumsum(gs-lambda)),0);
        
        
        g_max = gs./lambda;
        dual_scaling = max(1, max(g_max));
        
        theta = r/dual_scaling;
        f_dual = (theta'*theta) / 2;
        
        % Compute primal and dual objective
        objPrimal = f + lambda'*ys;
        objDual_scaled = -f_dual - theta'*b;
        objDual = -f - r'*b;
        
        Gap_scaled = objPrimal - objDual_scaled;
        Gap = objPrimal - objDual;
        absGap = abs(Gap);
        
        % Format string
        if (verbosity > 0)
            str = sprintf('   %9.2e  %9.2e  %9.2e',Gap, infeas/lambda(1), absGap / max(1,objPrimal));
        end
        
        
        % Check primal-dual gap
        if ((absGap/max(1,objPrimal) < tolRelGap) && ...
                (infeas < tolInfeas * lambda(1)))
            status = STATUS_OPTIMAL;
        end
    else
        str = '';
    end
    
    if (verbosity > 0)
        if ((verbosity == 2) || ...
                ((verbosity == 1) && (mod(iter,optimIter) == 0)))
            fprintf(fid,'%5d  %9.2e%s\n', iter,f,str);
        end
    end
    
    
    % Stopping criteria
    if (status == 0)
        if (iter >= iterations)
            status = STATUS_ITERATIONS;
        end
    end
    
    if (status ~= 0)
        if (verbosity > 0)
            fprintf(fid,'Exiting with status %d -- %s\n', status, STATUS_MSG{status});
        end
        break;
    end
    
    index_tmp = active;
    % Conduct the screening
    if ((mod(iter, screenIter) == 0))
        left = abs(g)/dual_scaling + sqrt(2*Gap_scaled)*l2;
        right = lambda(curr);
        index = find(left > right);
        curr = size(index,1);
        
        % Iterative strategy
        while (1)
            right = lambda(curr);
            index = find(left > right);
            size_tmp = curr;
            curr = size(index,1);
            if size_tmp == curr
                break;
            end
        end
        %
        
        % Eliminating inactive variables
        if curr < prev
            for i = 1:curr  %new
                active(i) = index_tmp(index(i));
            end
            x = x(index);
            g = g(index);
            A = A(:,index);
            l2 = l2(index);
            lambda = lambda(1:curr);
        end
    end
    
    nnz(iter) = curr;
    prev = curr;
    x_tilde = x;
    g_tilde = g/n;
    
    % Proximal SVRG update
    for p = 1:P
        j = randsample(1:n,batch);
        Delta_Gj = iGradF(A,x,x_tilde,j);
        
        v = Delta_Gj/batch + g_tilde;
        % Compute prox mapping
        x = proxFunction(x - gamma*v*n, lambda*gamma);
        Aprods = Aprods + 1;
    end
    
    %x_tilde = sum_x/P;
    x_tilde = x;
    Ax = A*x_tilde;
    
end


% Information structure
info = struct();
if (nargout > 1)
    info.runtime   = cputime - t0;
    info.Aprods    = Aprods + ceil(iter / gradIter);
    info.ATprods   = ATprods + iter;
    info.objPrimal = objPrimal;
    info.objDual   = objDual;
    info.infeas    = infeas;
    info.status    = status;
    info.L         = L;
    info.nnz       = nnz;
    info.x         = x;
    info.active    = active(1:curr);
end

end % Function Adlas


% ------------------------------------------------------------------------
function opt = getDefaultField(data,field,default)
% ------------------------------------------------------------------------
if isfield(data,field)
    opt = data.(field);
else
    opt = default;
end
end

