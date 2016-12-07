function[vals] = evaluate_obj_fun(X, y, Ws, lambda) 
    L = size(Ws, 1);
    N = size(X, 1);
    vals = zeros(L, 1);
    for i = 1:L
        vals(i) = norm(y - X*Ws(i, :)') / N + lambda * norm(Ws(i, :));
    end
end