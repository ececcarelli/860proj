N = 1000;
sig_X = 5;
sig_eps = 4;
W_true = [10, -20, 50, -20, 10];
lambda = 0.;
t0 = N/10;
epochs = 200;
m = N / 2;

[X, y, opt] = make_benchmark(N, W_true, sig_X, sig_eps, lambda, t0, epochs, m);

W_opt = ((X'*X + N*lambda * eye(size(W_true,2))) \ X'*y)' % true value
cfr_sgd = rls_sgd(X, y, opt); cfr_sgd.W'
cfr_svrg = rls_svrg(X, y, opt); cfr_svrg.W'
cfr_saga = rls_saga(X, y, opt); cfr_saga.W'

Ws = cfr_saga.Ws;
hold on;
plot(log(evaluate_obj_fun(X, y, cfr_sgd.Ws, 0)))
plot(log(evaluate_obj_fun(X, y, cfr_svrg.Ws, 0)))
plot(log(evaluate_obj_fun(X, y, cfr_saga.Ws, 0)))




