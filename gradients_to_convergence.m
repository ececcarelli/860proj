function[T, gcount] = gradients_to_convergence(X, y, cfr, tol)
    w_opt = ((X' * X) \ (X' * y))';
    min_err = evaluate_obj_fun(X, y, w_opt, 0);
    log_errs = log(evaluate_obj_fun(X, y, cfr.Ws, 0) - min_err);
    T = find(log_errs < tol,1);
    gcount = cfr.gcounts(T);
end
