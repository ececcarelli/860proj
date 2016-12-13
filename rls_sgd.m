function [cfr] = rls_sgd(X, y, opt)
% rls_pegasos(X, y, opt)
% computes a classifier for the primal formulation of RLS.
% The optimization is carried out using a stochastic gradient descent algorithm.
% The regularization parameter is set to the one found in opt.paramsel (set by the paramsel_* routines).
% In case of multiclass problems, the regularizers need to be combined with the opt.singlelambda function.
%
% INPUTS:
% -OPT: structure of options with the following fields:
%   fields that need to be set through previous gurls tasks:
%		- paramsel.lambdas (set by the paramsel_* routines)
%       - epochs
%   fields that need to be added by hand
%       -Xte
%       -yte
%       -t0
%   fields with default values set through the defopt function:
%		- singlelambda
% 
%   For more information on standard OPT fields
%   see also defopt
% 
% OUTPUT: structure with the following fields:
% -W: matrix of coefficient vectors of rls estimator for each class
% -t0: stepsize parameter 
% -W_sum: sum of the classifiers across iterations
% -count: number of iterations
% -acc_last: accuracy of the solution computed in the last iteration
% -acc_avg: average accuracy across iterations

[n,d] = size(X);

T = size(y,2);

opt.newprop('cfr', struct());
opt.cfr.W = zeros(d,T);
opt.cfr.W_sum = zeros(d,T);
opt.cfr.count = 0;
opt.cfr.t0 = opt.t0;
opt.cfr.gcount = 0;
opt.cfr.acc_last = [];
opt.cfr.acc_avg = [];
lambda = opt.singlelambda(opt.paramsel.lambdas);

L = opt.epochs * n;
opt.cfr.Ws = zeros(L, d);
opt.cfr.gcounts = zeros(L, 1);

% Run mulitple epochs
for i = 1:opt.epochs
	opt.cfr = rls_sgd_singlepass(X, y, opt);
    W = opt.cfr.W_sum/opt.cfr.count;
end	
cfr = opt.cfr;
cfr.W = opt.cfr.W_sum/opt.cfr.count;
