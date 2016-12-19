N = 1000;
sig_X = 50;
sig_eps = 10;
W_true = [10, -20, 50, -20, 10];
W_true = randn(1,5) + ones(1,5);
lambda = 0.;
% t0 = N/10;
epochs = 100;
m = N / 2;
% t0 =2.3368e+04;
t0 = 2.5131e+06;
[X, y, opt] = make_benchmark(N, W_true, sig_X, sig_eps, lambda, t0, epochs, m);

W_opt = ((X'*X + N*lambda * eye(size(W_true,2))) \ X'*y)' % true value

cfr_sgd = rls_sgd(X, y, opt); cfr_sgd.W'
cfr_svrg = rls_svrg(X, y, opt); cfr_svrg.W'
% cfr_svrgbb = rls_svrgbb(X, y, opt); cfr_svrgbb.W'
% opt.t0 = 5*N;
cfr_saga = rls_saga(X, y, opt); cfr_saga.W'
sgd_obj_fun = evaluate_obj_fun(X, y, cfr_sgd.Ws, 0);
svrg_obj_fun = evaluate_obj_fun(X, y, cfr_svrg.Ws, 0);
% svrgbb_obj_fun = evaluate_obj_fun(X, y, cfr_svrgbb.Ws, 0);
saga_obj_fun = evaluate_obj_fun(X, y, cfr_saga.Ws, 0);

min_obj_fun = evaluate_obj_fun(X, y, W_opt, 0);

% figure(1);
% plot(log(sgd_obj_fun));
% figure(2);
% plot(log(svrg_obj_fun));
% figure(3);
% plot(log(saga_obj_fun));

figure;
hold on;
plot(max(real(log(sgd_obj_fun - min_obj_fun)), 0));
plot(max(real(log(svrg_obj_fun - min_obj_fun)), 0));
plot(max(real(log(saga_obj_fun - min_obj_fun)), 0));
hold off;

% min_obj_fun = evaluate_obj_fun(X, y, W_opt, 0);
% opt.t0 = 5*N;
% % figure;
% % hold on;
% N_max = 5;
% t0s = [35, 40, 45, 50] * N;
% Ts = zeros(1, 1);
% 
% loops = 5;
% for i = 1:size(t0s, 2)
%     opt.t0 = t0s(i);
%     sumT = 0;
%     for j = 1:loops
%         c = rls_saga(X, y, opt); 
%         of = evaluate_obj_fun(X, y, c.Ws, 0);
%         [T, gc] = gradients_to_convergence(X, y, c, -35);
%         sumT = sumT + T;
%     end
%     Ts(i) = sumT / loops;
% %     plot(real(log(of - min_obj_fun)));
% end
% % hold off;
% figure;
% plot(t0s, Ts);
% 
% 
% 
% 
% 
