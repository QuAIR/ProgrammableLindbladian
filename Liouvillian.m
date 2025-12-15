function L = Liouvillian(d, H, L_ops)
        
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

    L = L_H_mat + L_D_mat;
end