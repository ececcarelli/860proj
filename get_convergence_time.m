function conv_time = get_convergence_time(scores, eps)
conv_time = -1;
for i=2:size(scores,1)
    if abs(scores(i-1,2) - scores(i,2)) <= eps
        conv_time = i-1;
        break
    end
end
        

end