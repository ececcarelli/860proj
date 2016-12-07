function[rslt] = plot_learning_curves(X,y,opt,algorithms,max_epochs,num_trials,title_str,use_log)
    rslt = cell(size(algorithms,2), 1);
    figure
    for i =1:size(algorithms,2)
        [scores, cfr] = learning_curve_input(X, y, opt,algorithms(i),max_epochs, num_trials);
        rslt{i} = cfr;
        if use_log
            plot(scores(:,1),log(scores(:,2)))
        else
            plot(scores(:,1),scores(:,2))
        end
        hold on;
    end

    title(title_str)
    xlabel('Iterations (epochs)') % x-axis label
    ylabel('Function value') % y-axis label
    legend(algorithms,'Location','northeast')
end

