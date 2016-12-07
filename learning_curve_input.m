function [scores, cfrs] = learning_curve_input(X, y, opt,algorithm,max_epochs, num_trials)
scores = zeros(max_epochs,2);
opt.epochs = max_epochs;
cfrs = cell(num_trials, 1);
for m=1:num_trials
    if strcmp(algorithm,'sgd')
        cfr = rls_sgd(X,y,opt);
    elseif strcmp(algorithm,'svrg')
        cfr = rls_svrg(X,y,opt);
    elseif strcmp(algorithm,'saga')
        cfr = rls_saga(X,y,opt);
    end  
    cfrs{m} = cfr;
%     size(scores)
%     size(cfr.scores/num_trials)
    scores = scores + cfr.scores/num_trials;
end
    

    
        
    



end

