function [Y, problem, info] = maxcut_manopt(A, p, Y0, tolgradnorm, tolreldualgap)

    n = size(A, 1);
    L = spdiags(sum(A, 2), 0, n, n) - A;

    assert(size(A, 2) == n && nnz(A-A') == 0, 'A must be symmetric.');
    
    if ~exist('p', 'var') || isempty(p)
        p = ceil(sqrt(8*n+1)/2);
    end

    manifold = obliquefactory(p, n, true);
    
    problem.M = manifold;
    

    % Products with A dominate the cost, hence we store the result.
    function store = prepare(Y, store)
        if ~isfield(store, 'AY')
            AY = A*Y;
            store.AY = AY;
            store.diagAYYt = sum(AY .* Y, 2);
        end
    end
    
    % Define the cost function to be /minimized/.
    problem.cost = @cost;
    function [f, store] = cost(Y, store)
        store = prepare(Y, store);
        f = .5*sum(store.diagAYYt);
    end

    % Define the Riemannian gradient.
    problem.grad = @grad;
    function [G, store] = grad(Y, store)
        store = prepare(Y, store);
        % G = store.AY - bsxfun(@times, Y, store.diagAYYt);
        G = store.AY - Y.*store.diagAYYt;
    end

    function flag = issquare(x)
        flag = (mod(sqrt(x), 1) == 0);
    end

    % If you want to, you can specify the Hessian as well.
    problem.hess = @hess;
    function [H, store] = hess(Y, Ydot, store)
        store = prepare(Y, store);
        % SYdot = A*Ydot - bsxfun(@times, Ydot, store.diagAYYt);
        SYdot = A*Ydot - Ydot.*store.diagAYYt;
        H = manifold.proj(Y, SYdot);
    end

    function [rel_duality_gap] = checkdualitygap(Y, iter)
        Yprime = Y ./ sqrt(sum(Y.^2, 2));
        S = A - spdiags(sum((A*Yprime).*Yprime, 2), 0, n, n); 
        eig_iter = ceil(10 * iter * log(n));
        [~, x0, ~] = ApproxMinEvecLanczosSE(S, n, eig_iter);

        maxcut_obj = sum(sum((L*Y).*Y))/4;
        duality_gap = -min(x0, 0) * n / 4;
        rel_duality_gap = duality_gap / maxcut_obj;
    end

    function stopnow = mystopfun(problem, x, info, last)
        stopnow = (info(last).iter > 5 && issquare(info(last).iter) && (checkdualitygap(x, 20) < tolreldualgap)); 
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


    if ~exist('Y0', 'var') || isempty(Y0)
        Y0 = [];
    end

    % Call your favorite solver.
    opts = struct();
    opts.verbosity = 2;
    opts.maxinner = 500;
    opts.stopfun = @mystopfun;
    if exist('tolgradnorm', 'var') && ~isempty(tolgradnorm)
        opts.tolgradnorm = tolgradnorm;
    end
    [Y, ~, info] = trustregions(problem, Y0, opts);
end
