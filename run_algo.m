function [cfrs] = run_algo(X, y, opt, algorithm, max_epochs, num_trials)
opt.epochs = max_epochs;
cfrs = cell(num_trials, 1);
for m=1:num_trials
    if strcmp(algorithm,'sgd')
%         opt.epochs = max_epochs * opt.m / size(X, 1);
        cfr = rls_sgd(X,y,opt);
    elseif strcmp(algorithm,'gd')
        cfr = rls_gd(X,y,opt);
    elseif strcmp(algorithm,'svrg')
        cfr = rls_svrg(X,y,opt);
    elseif strcmp(algorithm,'saga')
%         opt.epochs = max_epochs * opt.m / size(X, 1);
        cfr = rls_saga(X,y,opt);
    elseif strcmp(algorithm,'svrgbb')
%         opt.epochs = max_epochs * opt.m / size(X, 1);
        cfr = rls_svrgbb(X,y,opt);
    end  
    cfrs{m} = cfr;
%     size(scores)
%     size(cfr.scores/num_trials)
%     scores = scores + cfr.scores/num_trials;
end
    

    
        
    



end

