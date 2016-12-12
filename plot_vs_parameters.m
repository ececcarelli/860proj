function plot_vs_parameters(X, y, algorithms,param_name, param_start, param_end, tick_num, given_epochs, given_lambda, given_t0, given_m, num_trials, convergence_eps,use_log,title_str)
figure
for i =1:size(algorithms,2)
    [scores, cfrs, yAxis, xAxis] = hp_plot_input(X, y, algorithms(i),param_name, param_start, param_end, tick_num, given_epochs, given_lambda, given_t0, given_m, num_trials, convergence_eps);
    if use_log
        plot(scores(:,1),log(scores(:,2)))
    else
        plot(scores(:,1),scores(:,2))
    end
    hold on;
end

title(title_str)
xlabel(xAxis) % x-axis label
ylabel(yAxis) % y-axis label
legend(algorithms,'Location','northeast')