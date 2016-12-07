function [X, y, opt] = make_benchmark(N, W_true, sig_X, sig_eps, lambda, t0, epochs, m)
    d = size(W_true, 2);
    
    X = normrnd(zeros(N, d), sig_X);
    noise = normrnd(zeros(N, 1), sig_eps);
    y = X*W_true' + noise;

    opt = gurls_defopt('standard');
    lambdas = paramsel_loocvprimal(X, y, opt);
    if lambda==-1
        opt.paramsel.lambdas = lambdas.lambdas;
    else
        opt.paramsel.lambdas = lambda;
    end
    opt.newprop('t0',t0);
    opt.newprop('epochs',epochs);
    opt.newprop('m', m);
end