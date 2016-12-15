function [rslt] = act_plot_learning_curves(X,y,opt,algorithms,max_epochs,num_trials,title_str,use_log)
w_opt = ((X' * X) \ (X' * y))';
min_err = evaluate_obj_fun(X, y, w_opt, 0);
rslt = cell(size(algorithms,2), 1);
figure
for i =1:size(algorithms,2)
    cfr = run_algo(X, y, opt,algorithms(i),max_epochs, num_trials);
    rslt{i} = cfr;
    if use_log
%         plot(log(evaluate_obj_fun(X, y, rslt{i}{1}.Ws, 0) - min_err))
        plot(rslt{i}{1}.gcounts,log10(evaluate_obj_fun(X, y, rslt{i}{1}.Ws, 0) - min_err))
    else
%         plot(evaluate_obj_fun(X, y, rslt{i}{1}.Ws, 0) - min_err)
        plot(rslt{i}{1}.gcounts,evaluate_obj_fun(X, y, rslt{i}{1}.Ws, 0) - min_err)
    end
    hold on;
end


title(title_str)
xlabel('Number of single gradient computations') % x-axis label
ylabel('Function suboptimality') % y-axis label
% xlim([0, max([rslt{1}{1}.gcount,rslt{2}{1}.gcount,rslt{3}{1}.gcount])])
legend(algorithms,'Location','northeast')

min([rslt{1}{1}.gcount,rslt{2}{1}.gcount,rslt{3}{1}.gcount])
max([rslt{1}{1}.gcount,rslt{2}{1}.gcount,rslt{3}{1}.gcount])

end

