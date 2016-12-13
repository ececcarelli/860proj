function [rslt] = act_plot_learning_curves(X,y,opt,algorithms,max_epochs,num_trials,title_str,use_log)
w_opt = ((X' * X) \ (X' * y))';
min_err = evaluate_obj_fun(X, y, w_opt, 0);
rslt = cell(size(algorithms,2), 1);
for i =1:size(algorithms,2)
    cfr = run_algo(X, y, opt,algorithms(i),max_epochs, num_trials);
    rslt{i} = cfr;
    if use_log
        plot(log(evaluate_obj_fun(X, y, rslt{i}{1}.Ws, 0) - min_err))
    else
        plot(evaluate_obj_fun(X, y, rslt{i}{1}.Ws, 0) - min_err)
    end
end
figure
hold on;
plot(log(evaluate_obj_fun(X, y, rslt{1}{1}.Ws, 0) - min_err))
plot(log(evaluate_obj_fun(X, y, rslt{2}{1}.Ws, 0) - min_err))
plot(log(evaluate_obj_fun(X, y, rslt{3}{1}.Ws, 0) - min_err))
hold off;

title(title_str)
xlabel('Iterations (epochs)') % x-axis label
ylabel('Function value') % y-axis label
legend(algorithms,'Location','northeast')

end

