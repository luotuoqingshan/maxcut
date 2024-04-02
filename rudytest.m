function rudytest()
    
    % Store the results in "record": a 3-way array
    % where the idendxing goes:
    %   graph id ; performance metric ; algorithm
    record = zeros(81, 5, 2);

    for graphid = [67] %[1:67, 70, 72, 77, 81]

        use_mat_files = true;
        if ~use_mat_files

            data = importdata(sprintf('Gset/G%d', graphid), ' ', 1);
            header = sscanf(data.textdata{1}, '%d %d');
            n = header(1);
            m = header(2);
            I = data.data(:, 1);
            J = data.data(:, 2);
            W = data.data(:, 3);

            A = sparse([I;J], [J;I], [W;W], n, n, 2*m);

            save(sprintf('Gset/g%d.mat', graphid), 'A', 'n', 'm');

        else

            load(sprintf('Gset/g%d.mat', graphid), 'A', 'n', 'm'); %#ok<NASGU> 

        end

        %A = load('~/datasets/graphs/ca-AstroPh.mat').A_abs;
        
        A = abs(A); % Make sure the matrix is non-negative
        n = size(A, 1);
        m = nnz(A)/2;

        fprintf('Graph id: %d,\tn: %5d,\tm: %10d\n', graphid, n, nnz(A)/2);

        solvers = {@local_maxcut_manopt, ...
                   @local_maxcut_manopt_incremental};

        solver_names = {'Manopt         ', ...
                        'Manopt incr.   '};

        % Laplacian of the graph
        L = spdiags(sum(A, 2), 0, n, n) - A;
        
        for k = 1 : numel(solvers)

            % Skip CVX for large graphs
            % (it's super slow, and can crash for lack of memory)
            if (k == 5 && n >= 10000)
                record(graphid, :, k) = NaN;
                fprintf('\tSolver: %s\tlambdamin(S) in [%8g, %8g]\ttime: %8g\n', ...
                        solver_names{k}, NaN, NaN, NaN);
                continue;
            end
            
            solver = solvers{k};

            % Call the solver on the graph
            [Y, time, iter] = solver(L, A, 1e-2, 1e-2);
            fprintf('Number of iterations taken: %d\n', iter);

            [low, up, approx_min_eigval, obj, rel_duality_gap] = postprocessing(L, A, Y, iter);

            fprintf('\tSolver: %s\tlambdamin(S) in [%8g, %8g]\t Approx Min Eigval:%8g \t time: %8g, obj: %8g\t, relative duality gap: %8g\n', ...
                    solver_names{k}, low, up, approx_min_eigval, time, obj, rel_duality_gap);

            record(graphid, 1, k) = sum(sum((L*Y).*Y))/4;
            record(graphid, 2, k) = low;
            record(graphid, 3, k) = up;
            record(graphid, 4, k) = time;
            record(graphid, 5, k) = rel_duality_gap;
            
        end
        
    
        save rudytest.mat record;
    
%     
% 
%     % Extract n cuts from Y and compute the values via the Laplacian.
%     best_cut = -inf;
%     L = spdiags(sum(A, 2), 0, n, n) - A;
% %     for repeat = 1 : n
% %         x = sign(Y*randn(size(Y, 2), 1));
% %         cut_value = x'*L*x/4;
% %         if cut_value > best_cut
% %             best_cut = cut_value;
% %         end
% %     end
% 
%     S = A - spdiags(sum((A*Y).*Y, 2), 0, n, n);
%     % eigs(S, 6, 'SA')
%     % eigs(S, 6, 'LA')
%     [V, D] = mineig_manopt(S, 1);
%     
%     fprintf('Best cut: %d,\tbound: %.12g,\tlambdamin(S) = %g\ttime: %g\n', best_cut, trace(Y'*L*Y)/4, D, time);% \tconstraints error: %g, norm(sum(Y.*Y, 2)-1));
% 
%     % [Ydot, lambda] = hessianextreme(problem, Y, 'min');
%     % lambda
        
    end

end


function [low, up, approx_min_eigval, obj, rel_duality_gap] = postprocessing(L, A, Y, iter)
    % Normalize the rows to ensure constraint satisfaction.
    % Y = bsxfun(@times, Y, 1./sqrt(sum(Y.^2, 2)));
    % Newer versions of Matlab accept this code instead:
    Y = Y ./ sqrt(sum(Y.^2, 2));

    % Compute lambda min of S
    n = size(A, 1);
    S = A - spdiags(sum((A*Y).*Y, 2), 0, n, n);
    %[~, D] = mineig_manopt(S, 1);
    [low, up] = lambdamin(S, 1e-2);
    eig_iter = ceil(10 * iter * log(n));
    [~, approx_min_eigval, ~] = ApproxMinEvecLanczosSE(S, n, eig_iter);

    %fprintf('Approx Min Eigval: %8g \t, lambdamin(S) in [%8g, %8g]\n', approx_min_eigval, low, up);

    obj = sum(sum((L*Y).*Y))/4;
    duality_gap = -min(low, 0) * n / 4;
    rel_duality_gap = duality_gap / obj;
end


% Solve with Manopt
function [Y, time, iter] = local_maxcut_manopt(L, A, tolgradnorm, tolreldualitygap)    
    t = tic;
    [Y, ~, info] = maxcut_manopt(A, [], [], tolgradnorm, tolreldualitygap);
    iter = info(end).iter;
    time = toc(t);
end

% Solve with Manopt incremental
function [Y, time, iter] = local_maxcut_manopt_incremental(L, A, tolgradnorm, tolreldualitygap)
    t = tic;
    [Y, info, iter] = maxcut_manopt_incremental(A, tolgradnorm, tolreldualitygap);
    time = toc(t);
end


%% Lanczos method storage efficeint implementation
function [v, xi, i] = ApproxMinEvecLanczosSE(M, n, q)
    % Approximate minimum eigenvector
    % Vanilla Lanczos method

    q = min(q, n-1);                    % Iterations < dimension!

    if isnumeric(M), M = @(x) M*x; end

    aleph = zeros(q,1);                 % Diagonal Lanczos coefs
    beth = zeros(q,1);                  % Off-diagonal Lanczos coefs

    v = randn(n, 1);                   % First Lanczos vector is random
    v = v / norm(v);
    vi = v;

    % First loop is to find coefficients
    for i = 1 : q

        vip1 = M ( vi );			% Apply M to previous Lanczos vector
        aleph(i) = real(vi' * vip1);		% Compute diagonal coefficients
        
        if (i == 1)                     % Lanczos iteration
            vip1 = vip1 - aleph(i) * vi;
        else
            vip1 = vip1 - aleph(i) * vi - beth(i-1) * vim1;
        end
        
        beth(i) = norm( vip1 );            % Compute off-diagonal coefficients
        
        if ( abs(beth(i)) < sqrt(n)*eps ), break; end
        
        vip1 = vip1 / beth(i);        % Normalize
        
        vim1 = vi;  % update
        vi = vip1;
        
    end

    % i contains number of completed iterations
    B = diag(aleph(1:i), 0) + diag(beth(1:(i-1)), +1) + diag(beth(1:(i-1)), -1);
    [U, D] = cgal_eig(0.5*(B+B'));
    [xi, ind] = min(D);
    Uind1 = U(:,ind);

    % Second loop is to find compute the vector (on the fly)
    aleph = zeros(q,1);                 % Diagonal Lanczos coefs
    beth = zeros(q,1);                  % Off-diagonal Lanczos coefs
    vi = v;
    v = zeros(n,1);
    for i = 1 : length(Uind1)

        v = v + vi*Uind1(i);

        vip1 = M ( vi );                 % Apply M to previous Lanczos vector
        aleph(i) = real(vi' * vip1);		% Compute diagonal coefficients
        
        if (i == 1)                     % Lanczos iteration
            vip1 = vip1 - aleph(i) * vi;
        else
            vip1 = vip1 - aleph(i) * vi - beth(i-1) * vim1;
        end
        
        beth(i) = norm( vip1 );    % Compute off-diagonal coefficients
        
        % if ( abs(beth(i)) < sqrt(n)*eps ), break; end
        
        % if i >= numit, warning('numerical error in Lanczos'); break; end
        
        vip1 = vip1 / beth(i);          % Normalize
        
        vim1 = vi;  % update
        vi = vip1;
            
    end

    i = 2*i; % we looped twice

    % Next lines are unnecessary in general, but I observed numerical errors in
    % norm(v) at some experiments, so let's normalize it for robustness. 
    nv = norm(v);
    xi = xi*nv;
    v = v/nv;
end

function [V,D] = cgal_eig(X)
    % Eig in Lanczos based LMO solver sometimes fall into numerical issues. 
    % This function replaces eig with a SVD based solver, in case eig does not
    % converge. 
    try
        [V,D]       = eig(X,'vector');
    catch 
        warning('eig did not work. Using the svd based replacement instead.');
        [V,D,W]     = svd(X);
        D           = diag(D).' .* sign(real(dot(V,W,1)));
        [D,ind]     = sort(D);
        V           = V(:,ind);
    end
end