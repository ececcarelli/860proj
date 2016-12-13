function [cfr] = rls_svrg_singlepass(X, y, opt)
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
W_last = cfr.W_last;
W_sum = cfr.W_sum;
count = cfr.count;
t0 = cfr.t0;
m = cfr.m;
Ws = cfr.Ws;

mu = zeros(d,1); % average of gradients 
for idx = 1:n
    mu = mu + rls_grad(W, X, y, lambda, idx); 
end
mu = mu / n;

%% Initialization
iter = 0;
seq = randperm(m); 
while iter < m,
    iter = iter + 1;
    idx = seq(iter);
    
    %% Stepsize
    eta = 1.0/(t0); % decaying step size
    
    %% Update Equations
    W = W - eta * (rls_grad(W, X, y, lambda, idx) - rls_grad(W_last, X, y, lambda, idx) + mu); % CHECK SIGN ON ETA

    %% Averaging
    W_sum = W_sum + W;
    count = count + 1;
    
    %% Update tables
    Ws(count, :) = W;
    
end
cfr.W = W;
cfr.Ws = Ws;
cfr.W_last = W;
cfr.W_sum = W_sum;
cfr.count = count;
cfr.iter = iter;
cfr.C = [];
cfr.X = [];
end


function[g] = rls_grad(W, X, y, lambda, idx)
    xt = X(idx,:);
    r = y(idx,:); 
    g = 2 * (xt'*(xt*W - r) + lambda*W);
end



