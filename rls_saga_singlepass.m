function [cfr] = rls_saga_singlepass(X, y, opt)
% rls_sgd_singlepass(X,y,OPT)
% utility function called by rls_svrg
% computes a single pass for sgd algorithm, performing the 
% gradient descent over a batch of training samples.
%
% INPUTS:
% -X: input data matrix
% -y: labels matrix
% -OPT: structure of options with the following fields:
%   fields that need to be set through previous gurls tasks:
%		- paramsel.lambdas (set by the paramsel_* routines)
%       - epochs
%   fields that need to be added by hand
%       -Xte
%       -yte
%       -m (update frequency)
%       -t0
%   fields with default values set through the defopt function:
%		- singlelambda
% 
%   For more information on standard OPT fields
%   see also defopt
% 
% OUTPUT: structure with the following fields:
% -W: matrix of coefficient vectors of rls estimator for each class
% -W_sum: sum of the classifiers across iterations
% -t0: stepsize parameter
% -m: update frequency
% -count: number of iterations
% -acc_last: accuracy of the solution computed in the last iteration
% -acc_avg: average accuracy across iterations

lambda = opt.singlelambda(opt.paramsel.lambdas);

%% Inputs
[n,d] = size(X); 
[T] = size(y,2);

%% Initialization
cfr = opt.cfr;

W = cfr.W;
u = cfr.u;
grad_table = cfr.grad_table;
W_sum = cfr.W_sum;
count = cfr.count;
t0 = cfr.t0;

Ws = cfr.Ws;

%% Initialize grad table
if cfr.count == 0,
    for idx = 1:n,
        grad_table(idx, 1:d) = rls_grad(W, X, y, lambda, idx); 
    end
end

%% Initialization
iter = 0;
while iter < n,
    iter = iter + 1;
    
    %% Stepsize
    eta = n; % decaying step size
    
    %% Update Equations
    W = u - 1/eta * sum(grad_table, 1)';
    u = u + (W - u) / n;
    idx = randi(n); % update random row of table
    grad_table(idx, 1:d) = rls_grad(W, X, y, lambda, idx); 
    
    %% Averaging
    W_sum = W_sum + W;
    count = count + 1;
    
    %% Update tables
    Ws(count, :) = W; 
    
end

cfr.W = W;
cfr.u = u;
cfr.grad_table = grad_table;
cfr.W_sum = W_sum;
cfr.count = count;
cfr.iter = iter;
cfr.Ws = Ws;
cfr.C = [];
cfr.X = [];
end


function[g] = rls_grad(W, X, y, lambda, idx)
    xt = X(idx,:);
    r = y(idx,:); 
    g = 2 * (xt'*(xt*W - r) + lambda*W);
end




