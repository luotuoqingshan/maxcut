function test_maxcut(varargin)
    
    p = inputParser;
    addOptional(p, 'seed', 0, @isnumeric);
    addOptional(p, 'tol', 0.01, @isnumeric);
    addOptional(p, 'graph', 'G1', @ischar);
    addOptional(p, 'solver', 'manopt', @ischar);
    addOptional(p, 'R', 0, @isnumeric);

    parse(p, varargin{:});
    seed = p.Results.seed; 
    tol = p.Results.tol;
    graphname = p.Results.graph;
    solver = p.Results.solver;
    R = p.Results.R;

    rng(seed,'twister');

    %% Modify these paths before running
    data = load(['data/', graphname, '.mat']);  

    A = data.A;
    
    %A = abs(A); % Make sure the matrix is non-negative
    % we have already done this step in preprocessing
    n = size(A, 1);

    fprintf('Graph id: %s,\tn: %5d,\tm: %10d\n', graphname, n, nnz(A)/2);

    % Laplacian of the graph
    L = spdiags(sum(A, 2), 0, n, n) - A;
    record = zeros(5, 1);
    
    % Call the solver on the graph
    if convertCharsToStrings(solver) == "manopt"
        [Y, time, iter, finaltolgradnorm] = local_maxcut_manopt_exp_decay(A, R, tol, 10.0, 5.0);
        solver_name = 'Manopt         ';
    end
    %else
    %    [Y, time, iter, finaltolgradnorm] = local_maxcut_manopt_incremental_exp_decay(A, tol, 10.0, 5.0);
    %    solver_name = 'Manopt incr.   ';
    %end
    fprintf('Number of iterations taken: %d\n', iter);

    [approx_min_eigval, obj, rel_duality_gap] = postprocessing(L, A, Y);

%   Extract 100 cuts from Y and compute the values via the Laplacian.
    best_cut = -inf;
    for repeat = 1 : 100 
        x = sign(Y*randn(size(Y, 2), 1));
        cut_value = x'*L*x/4;
        if cut_value > best_cut
            best_cut = cut_value;
        end
    end

    fprintf('\tSolver: %s\t R: %8g \tApprox Min Eigval:%8g \t time: %8g, obj: %8g\t cut: %8g\t relative duality gap: %8g\n', ...
            solver_name, R, approx_min_eigval, time, obj, best_cut, rel_duality_gap);

    record(1) = sum(sum((L*Y).*Y))/4; %objective value 
    record(2) = best_cut;             %best cut value
    record(3) = rel_duality_gap;      %relative suboptimality
    record(4) = time;                 %running time
    record(5) = finaltolgradnorm;     %the final tolerance for the gradient norm
    
    %% Modify these paths before running
    if ~exist(['output/',graphname, '/manopt', ],'dir') 
        mkdir(['output/',graphname, '/manopt']);
    end
    save(['output/', graphname, '/manopt/', solver, '-R-', num2str(R), '-seed-', num2str(seed), '-tol-', num2str(tol), '.mat'],'record','-v7.3');
end


function [approx_min_eigval, obj, rel_duality_gap] = postprocessing(L, A, Y)
    % Normalize the rows to ensure constraint satisfaction.
    % Y = bsxfun(@times, Y, 1./sqrt(sum(Y.^2, 2)));
    % Newer versions of Matlab accept this code instead:
    Y = Y ./ sqrt(sum(Y.^2, 2));

    % Approximate lambda min of S
    n = size(A, 1);
    S = A - spdiags(sum((A*Y).*Y, 2), 0, n, n);
    eig_iter = ceil(100 * log(n));
    [~, approx_min_eigval, ~] = ApproxMinEvecLanczosSE(S, n, eig_iter);


    obj = sum(sum((L*Y).*Y))/4;
    duality_gap = -min(approx_min_eigval, 0) * n / 4;
    rel_duality_gap = duality_gap / obj;
end


% Solve with Manopt
function [Y, iter] = local_maxcut_manopt(A, R, Y0, tolgradnorm)    
    [Y, ~, info] = maxcut_manopt(A, R, Y0, tolgradnorm);
    iter = info(end).iter;
end


function  [Y, time, totaliter, finaltolgradnorm] = local_maxcut_manopt_exp_decay(A, R, tolreldualgap, tolgradnorm0, tolgraphnormdecay)
    finaltolgradnorm = tolgradnorm0;
    tolgradnorm = tolgradnorm0;
    Y0 = [];
    t = tic;
    totaliter = 0;
    while true
        [Y, iter] = local_maxcut_manopt(A, R, Y0, tolgradnorm);
        totaliter = totaliter + iter;
        L = spdiags(sum(A, 2), 0, size(A, 1), size(A, 1)) - A;
        [approx_min_eigval, obj, reldualgap] = postprocessing(L, A, Y);
        if reldualgap < tolreldualgap
            break;
        end
        tolgradnorm = tolgradnorm / tolgraphnormdecay;
        finaltolgradnorm = tolgradnorm;
        Y0 = Y;
    end
    time = toc(t);
    fprintf('\tSolver: Manopt\t Approx Min Eigval:%8g \t time: %8g, obj: %8g\t, relative duality gap: %8g\n', ...
                approx_min_eigval, time, obj, reldualgap);
end

% Solve with Manopt incremental. The following code is not in a good shape
% so we comment them out. The main problem is that I don't know how to 
% choose good tolerance for different ranks, the current one keeps them 
% same but will result in a slower speed than Manopt w/o incremental. 

%function [Y, time, iter] = local_maxcut_manopt_incremental(A, tolgradnorm)
%    t = tic;
%    [Y, ~, iter] = maxcut_manopt_incremental(A, tolgradnorm);
%    time = toc(t);
%end
%
%function [Y, time, iter, finaltolgradnorm] = local_maxcut_manopt_incremental_exp_decay(A, tolreldualgap, tolgradnorm0, tolgraphnormdecay)
%    finaltolgradnorm = tolgradnorm0;
%    tolgradnorm = tolgradnorm0;
%    while true
%        [Y, time, iter] = local_maxcut_manopt_incremental(A, tolgradnorm);
%        L = spdiags(sum(A, 2), 0, size(A, 1), size(A, 1)) - A;
%        fprintf("Number of iterations: %8g\n", iter);
%        [approx_min_eigval, obj, reldualgap] = postprocessing(L, A, Y);
%        fprintf('\tSolver: Manopt incr\t Approx Min Eigval:%8g \t time: %8g, obj: %8g\t, relative duality gap: %8g\n', ...
%                    approx_min_eigval, time, obj, reldualgap);
%        if reldualgap < tolreldualgap
%            break;
%        end
%        tolgradnorm = tolgradnorm / tolgraphnormdecay;
%        finaltolgradnorm = tolgradnorm;
%    end
%end


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