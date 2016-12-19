function [cfr] = rls_sgd_singlepass(X, y, opt)
% rls_sgd_singlepass(X,BY,OPT)
% utility function called by rls_sgd
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
%       -t0 : stepsize parameter
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
W_sum = cfr.W_sum;
count = cfr.count;
t0 = cfr.t0;
Ws = cfr.Ws;
gcount = cfr.gcount;
gcounts = cfr.gcounts;

%% Initialization
iter = 0;

seq = randperm(n); 
while iter < n,
    iter = iter + 1;
    idx = seq(iter);
    
    %% Stepsize
    eta = 1.0/(count + t0); % decaying step size
    
    %% Update Equations
    xt = X(idx,:);

    r = y(idx,:); 
    W = W - eta*rls_grad(W, X, y, lambda, idx);
    gcount = gcount + 1;

    %% Averaging
    W_sum = W_sum + W;
    count = count + 1;
    
    %% Update tables
    Ws(count, :) = W;
    gcounts(count) = gcount;
    
end
cfr.W = W;
cfr.W_last = W;
cfr.W_sum = W_sum;
cfr.count = count;
cfr.gcount = gcount;
cfr.gcounts = gcounts;
cfr.iter = iter;
cfr.Ws = Ws;
cfr.C = [];
cfr.X = [];
end

function[g] = rls_grad(W, X, y, lambda, idx)
    xt = X(idx,:);
    r = y(idx,:); 
    g = 2*(xt'*(xt*W - r) + lambda*W);
end




