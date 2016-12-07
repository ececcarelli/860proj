function [scores, cfrs] = hp_plot_input(X, y, algorithm,param_name, param_start, param_end, tick_num, given_epochs, given_lambda, given_t0, given_m, num_trials, convergence_eps)
%param_names: lambda, spec_lambda, t0, m, epochs (use prev function)

opt = gurls_defopt('standard');
lambdas = paramsel_loocvprimal(X, y, opt);
if given_lambda==-1
    opt.paramsel.lambdas = lambdas.lambdas;
else
    opt.paramsel.lambdas = given_lambda;
end
opt.newprop('t0',given_t0);
opt.newprop('epochs',given_epochs);
opt.newprop('m', given_m);

if strcmp(param_name,'epochs')
    [scores, cfrs] = learning_curve_input(X, y, opt,algorithm,given_epochs, num_trials);
else

    if strcmp(param_name,'spec_lambda')
        param_start = param_start*opt.singlelambda(opt.paramsel.lambdas);
        param_end = param_end*opt.singlelambda(opt.paramsel.lambdas);
    end
    param_range = param_end - param_start;
    tick_size = param_range/tick_num;
    param_array = param_start + tick_size*(1:tick_num);
    
    
    cfrs = cell(num_trials, tick_num);

    scores = zeros(tick_num,2);
    for m=1:num_trials

        trial_scores = zeros(tick_num,2);
        for i=1:tick_num
            param_value = param_array(i);
            trial_scores(i,1) = param_value;
            if (strcmp(param_name,'lambda') || strcmp(param_name,'spec_lambda'))
                opt.paramsel.lambdas = param_value;
            elseif strcmp(param_name,'t0')
                opt.t0 = param_value;
            elseif strcmp(param_name,'m')
                opt.m = param_value;
            end


            if strcmp(algorithm,'sgd')
                cfr = rls_sgd(X,y,opt);
            elseif strcmp(algorithm,'svrg')
                cfr = rls_svrg(X,y,opt);
            elseif strcmp(algorithm,'saga')
                cfr = rls_saga(X,y,opt);
            end
            conv_time = get_convergence_time(cfr.scores, convergence_eps); 
            cfrs{m, i} = cfr;
            trial_scores(i,2) = conv_time;
        end
        scores = scores + trial_scores/num_trials;
    end


end

