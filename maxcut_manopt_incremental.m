function [Y, info, totaliter] = maxcut_manopt_incremental(A, tolgradnorm, tolreldualitygap)
% Simplistic incremental rank strategy for maxcut_manopt.
% Nicolas Boumal, June 10, 2016.

    n = size(A, 1);

    % Explore from rank p_min to rank p_max
    p_min = 2;
    p_max = ceil(sqrt(8*n+1)/2);
    p_steps = 5;
    pp = round(linspace(p_min, p_max, p_steps));
    Y0 = [];
    totaliter = 0;
    for k = 1 : length(pp)
        [Y, ~, info] = maxcut_manopt(A, pp(k), Y0, tolgradnorm, tolreldualitygap);
        totaliter = totaliter + info(end).iter;
        if k < length(pp)
            jmp = 1e-5; % arbitrary parameter value...
            Y0 = [Y jmp*randn(n, pp(k+1)-pp(k))];
            % Y0 = bsxfun(@times, Y0, 1./sqrt(sum(Y0.^2, 2)));
            Y0 = Y0 ./ sqrt(sum(Y0.^2, 2));
        end
    end
    
end
