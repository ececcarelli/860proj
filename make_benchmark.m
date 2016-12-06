function [X, y, opt] = make_benchmark(N, W_true, sig_array, t0, epochs)
d = size(W_true, 1);
s_n = size(sig_array);
X = zeros(N, d);
noise = zeros(N, 1);
for i=1:s_n
    sig_X = sig_array(i);
    X = X + normrnd(zeros(N, d), sig_X);
    noise = noise + normrnd(zeros(N, 1), sig_X);
end

y = X*W_true + noise;

opt = gurls_defopt('standard');
lambdas = paramsel_loocvprimal(X, y, opt);
opt.paramsel.lambdas = lambdas.lambdas;
opt.newprop('t0',t0);
opt.newprop('epochs',epochs);
opt.newprop('m', N/2);