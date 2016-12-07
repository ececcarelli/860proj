function scores = learning_curve_input(X, y, opt,algorithm,max_epochs, num_trials)
scores = zeros(max_epochs,2);
opt.epochs = max_epochs;
for m=1:num_trials
    if strcmp(algorithm,'sgd')
        cfr = rls_sgd(X,y,opt);
    elseif strcmp(algorithm,'svrg')
        cfr = rls_svrg(X,y,opt);
    elseif strcmp(algorithm,'saga')
        cfr = rls_saga(X,y,opt);
    end    
%     size(scores)
%     size(cfr.scores/num_trials)
    scores = scores + cfr.scores/num_trials;
end
    

    
        
    



end

