function [cfr] = rls_svrgbb_singlepass(X, y, opt)
% rls_sgd_singlepass(X,y,OPT)
% utility function called by rls_svrgbb
% computes a single pass for SVRG-BB algorithm.
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
%       -t0 : initial stepsize parameter
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
W_last = cfr.W_tilde;
G_last =cfr.grad;
W_sum = cfr.W_sum;
count = cfr.count;
gcounts = cfr.gcounts;
gcount = cfr.gcount;
t0 = cfr.t0;
m = cfr.m;
Ws = cfr.Ws;
etas = cfr.etas;

W_tilde = W;

G = zeros(d,1); % average of gradients 
for idx = 1:n
    G = G + rls_grad(W_tilde, X, y, lambda, idx); 
end
gcount = gcount + n;
G = G / n;

%% Stepsize
if count < 1
    eta = 1.0/(t0); 
else
    delta_G = G-G_last;
    eta = 1/m * (W_tilde - W_last)' * (W_tilde - W_last) / ((W_tilde - W_last)' * delta_G); % BB
    if log(abs(delta_G)) == -Inf  % if eta = -Inf, then use last step size
        eta = 0;%etas(floor(count / m));
    end
end
etas(floor(count / m) + 1) = eta;
%% Initialization
iter = 0;
seq = mod(randperm(m), n) + 1; 
while iter < m,
    iter = iter + 1;
    idx = seq(iter);
    
    %% Update Equations
    W = W - eta * (rls_grad(W, X, y, lambda, idx) - rls_grad(W_tilde, X, y, lambda, idx) + G); % CHECK SIGN ON ETA
    gcount = gcount + 2;
    
    %% Averaging
    W_sum = W_sum + W;
    count = count + 1;
    
    %% Update tables
    Ws(count, :) = W;
    gcounts(count) = gcount;
    
end
cfr.W = W;
cfr.Ws = Ws;
cfr.etas = etas;
cfr.W_last = W;
cfr.grad = G;
cfr.W_tilde = W_tilde;
cfr.W_sum = W_sum;
cfr.count = count;
cfr.gcount = gcount;
cfr.gcounts = gcounts;
cfr.iter = iter;
cfr.C = [];
cfr.X = [];
end


function[g] = rls_grad(W, X, y, lambda, idx)
    xt = X(idx,:);
    r = y(idx,:); 
    g = 2 * (xt'*(xt*W - r) + lambda*W);
end



