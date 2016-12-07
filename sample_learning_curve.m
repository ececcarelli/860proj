[X, y, opt] = make_benchmark(1000, [10,-20,50,-20,10], 5, 5, -1, 1000, 75, 500);
cfrs = plot_learning_curves(X, y, opt,{'sgd','saga', 'svrg'},70, 2, 'title',true);