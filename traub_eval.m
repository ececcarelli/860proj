[X, y, opt] = make_benchmark(1000, [10, -20, 30, -40, 50], 2, 1, -1, 1000, 65, 3000);
w_opt = ((X' * X) \ (X' * y))';
min_err = evaluate_obj_fun(X, y, w_opt, 0);
rslt = plot_learning_curves(X,y,opt,{'sgd', 'svrg', 'saga'},80,1,'Iterations vs Value of Optimized Function',true);
figure (2)
hold on;
plot(rslt{1}{1}.gcounts, log(evaluate_obj_fun(X, y, rslt{1}{1}.Ws, 0) - min_err))
plot(rslt{2}{1}.gcounts, log(evaluate_obj_fun(X, y, rslt{2}{1}.Ws, 0) - min_err))
plot(rslt{3}{1}.gcounts, log(evaluate_obj_fun(X, y, rslt{3}{1}.Ws, 0) - min_err))
xlim([0, min([rslt{1}{1}.gcount,rslt{2}{1}.gcount,rslt{3}{1}.gcount])])
hold off;