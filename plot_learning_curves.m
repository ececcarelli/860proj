function plot_learning_curves(X, y, opt,algorithms,max_epochs, num_trials, titler,use_log)
figure
for i =1:size(algorithms,2)
    scores = learning_curve_input(X, y, opt,algorithms(i),max_epochs, num_trials);
    if use_log
        plot(scores(:,1),log(scores(:,2)))
    else
        plot(scores(:,1),scores(:,2))
    end
    hold on;
end
    
title(titler)
xlabel('Iterations (epochs)') % x-axis label
ylabel('Function value') % y-axis label
legend(algorithms,'Location','northeast')


end

