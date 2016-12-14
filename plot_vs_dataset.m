function plot_vs_dataset(algorithms, param, N,d,magnitude, sig_x, sig_eps, param_start, param_end, tick_num, given_epochs, given_lambda, convergence_eps,title_str)

param_range = param_end - param_start;
tick_size = param_range/tick_num;
param_array = param_start + tick_size*(1:tick_num);
scores = zeros(tick_num,1+size(algorithms,2));
if (~strcmp(param, 'd'))
    base_W = randn(1,d) + ones(1,d);
end

for j=1:tick_num
    scores(j,1) = param_array(j);
    if strcmp(param, 'n')
        N = param_array(j);
        W_true = base_W*magnitude;
        [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, N, given_epochs, 2*N);
    elseif strcmp(param, 'd')
        W_true = magnitude*(ones(1,d)+randn(1,ceil(param_array(j))));
        [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, N, given_epochs, 2*N);
    elseif strcmp(param, 'magnitude')
        W_true = base_W*param_array(j);
        [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, N, given_epochs, 2*N);
    elseif strcmp(param, 'sig_x')
        sig_x = param_array(j);
        W_true = base_W*magnitude;
        [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, N, given_epochs, 2*N);
    elseif strcmp(param, 'sig_eps')
        sig_eps = param_array(j);
        W_true = base_W*magnitude;
        [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, N, given_epochs, 2*N);
    end
    for i =1:size(algorithms,2)
        if strcmp(algorithms(i),'sgd')
            cfr = rls_sgd(X,y,opt);
        elseif strcmp(algorithms(i),'svrg')
            cfr = rls_svrg(X,y,opt);
        elseif strcmp(algorithms(i),'saga')
            cfr = rls_saga(X,y,opt);
        end
        [~, gcount] = gradients_to_convergence(X, y, cfr, convergence_eps); 
        if size(gcount,1) ==0
            gcount = -1;
        end
        scores(j,i+1) = gcount;
        scores
    end
end

for k=1:size(algorithms,2)
    plot(scores(:,1),scores(:,k+1));
    hold on;
end

title(title_str)
xlabel(['Value of ' param]) % x-axis label
ylabel('Gradient computations until convergence') % y-axis label
legend(algorithms,'Location','northeast')
end

