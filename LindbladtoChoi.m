function J = LindbladtoChoi(d, t, H, L_ops)

    %% check
    assert(d==size(H,1), 'dimension does not match');
        
    %% construct Liouvillian
    I_d = eye(d);

    % local evolution
    L_H_mat = -1i * (kron(H, I_d) - kron(I_d, conj(H)));

    % disspation 
    L_D_mat = zeros(d^2, d^2);
    for k = 1:length(L_ops)
        Lk = L_ops{k};
        Lkd_Lk = Lk' * Lk; % L_k dagger * L_k
        
        term1 = kron(Lk,conj(Lk));
        term2 = -0.5 * kron(I_d, Lkd_Lk);
        term3 = -0.5 * kron(Lkd_Lk.', I_d); %
        
        L_D_mat = L_D_mat + term1 + term2 + term3;
    end

    L_mat = L_H_mat + L_D_mat;

    % channel matrix
    K_t = expm(L_mat * t);

    %% compute Choi matrix
    % MES = d*MaxEntangled(d, 0, 1) * MaxEntangled(d, 0, 1)';
    % MES_T = MES.'
    % vec_J = kron(eye(d^2), K_t) * MES_T(:);

    J = 0;
    % construct the reshuffling
    for i = 1:d
        for j = 1:d
            % define matrix element E_ij
            E_ij = zeros(d,d);
            E_ij(i,j) = 1;
            
            % vectorize E_ij
            vec_E_ij = E_ij.';
            vec_E_ij = vec_E_ij(:);
            
            % acting with channel K_t
            vec_M_ij = K_t * vec_E_ij;
            M_ij = reshape(vec_M_ij, [d, d]);
            M_ij = M_ij.';
            
            % construct Choi matrix
            J = J + kron(E_ij, M_ij);
        end
    end

    J = J;
end