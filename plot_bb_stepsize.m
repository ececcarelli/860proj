function[rslt] = plot_bb_stepsize(t0s, N, d, magnitude, sig_x, sig_eps, given_epochs, given_lambda)
% Plots no. of gradient evaluataions until convergence within
% convergence_eps; 

base_W = randn(1,d) + ones(1,d);
% t0 = 2.3368e+04;

rslt = {};


for j=1:size(t0s, 2)
    t0 = t0s(j);
    W_true = base_W*magnitude;
    [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, t0, given_epochs, 3*N);
    cfr = rls_svrgbb(X,y,opt); 
    rslt{j} = cfr.etas;
end


figure
hold on;
strs = {};
for j=1:size(t0s, 2)
    plot(log10(rslt{j}));
    strs{j} = num2str(1/t0s(j), '%10.1e\n');
end

title('Log SVGR-BB Step Size for Various Initial Step Sizes')
xlabel(['Epochs']) % x-axis label
ylabel('Log Stepize') % y-axis label
legend(strs,'Location','northeast')
