function[scores] = plot_stepsize_v_convergence(algorithms, param, N, d, magnitude, sig_x, sig_eps, t0, param_start, param_end, tick_num, linear, given_epochs, given_lambda, convergence_eps ,title_str, num_tries)
% Plots no. of gradient evaluataions until convergence within
% convergence_eps; 
if linear
    param_array = linspace(param_start, param_end, tick_num);
else
    param_array = logspace(log10(param_start), log10(param_end), tick_num);
end
scores = zeros(tick_num,1+size(algorithms,2));
success = zeros(tick_num,1+size(algorithms,2));
if (~strcmp(param, 'd'))
    base_W = randn(1,d) + ones(1,d);
end

% t0 = 2.3368e+04;

for k=1:num_tries
    for j=1:tick_num
        scores(j,1) = param_array(j);
        if strcmp(param, 'n')
            N = param_array(j);
            W_true = base_W*magnitude;
            [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, t0, given_epochs, 2*N);
        elseif strcmp(param, 'd')
            W_true = magnitude*(ones(1,d)+randn(1,ceil(param_array(j))));
            [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, t0, given_epochs, 3*N);
        elseif strcmp(param, 'magnitude')
            W_true = base_W*param_array(j);
            [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, t0, given_epochs, 3*N);
        elseif strcmp(param, 'sig_x')
            sig_x = param_array(j);
            W_true = base_W*magnitude;
            [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, t0, given_epochs, 3*N);
        elseif strcmp(param, 'sig_eps')
            sig_eps = param_array(j);
            W_true = base_W*magnitude;
            [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, t0, given_epochs, 3*N);
        elseif strcmp(param, 't0')
            t0 = param_array(j);
            W_true = base_W*magnitude;
            [X, y, opt] = make_benchmark(N, W_true, sig_x, sig_eps, given_lambda, t0, given_epochs, 3*N);
        end
        for i =1:size(algorithms,2)
            if strcmp(algorithms(i),'sgd')
                cfr = rls_sgd(X,y,opt);
            elseif strcmp(algorithms(i),'svrg')
                cfr = rls_svrg(X,y,opt);
            elseif strcmp(algorithms(i),'saga')
                cfr = rls_saga(X,y,opt);
            elseif strcmp(algorithms(i),'svrgbb')
                cfr = rls_svrgbb(X,y,opt);
            end
            [~, gcount] = gradients_to_convergence(X, y, cfr, convergence_eps); 
            if gcount > 0
                s = success(j,i+1) + 1;
                success(j,i+1) = s;
                scores(j,i+1) = scores(j,i+1) * (s - 1) / s + gcount / s;
            end
            scores
        end
    end
end

success

for j=1:tick_num
    for i =1:size(algorithms,2)
        if success(j,i+1) == 0
            scores(j,i+1) = nan;
        end
    end
end

scores
    


figure
if linear
    for k=1:size(algorithms,2)
        plot(scores(:,1),scores(:,k+1));
        hold on;
    end
    xlabel(['Value of ' param]) % x-axis label
else
    for k=1:size(algorithms,2)
        plot(log(scores(:,1)),scores(:,k+1));
        hold on;
    end
    xlabel(['Value of log ' param ]) % x-axis label
end
    

title(title_str)
% xlabel(['Value of ' param]) % x-axis label
ylabel('Gradient computations until 1e-12 suboptimality') % y-axis label
legend(algorithms,'Location','northeast')
end

