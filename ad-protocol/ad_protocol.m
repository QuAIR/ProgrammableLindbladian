clear; clc;


%% model setup 
d = 2;
sample = 1000;
T = 1;
I = eye(d);
X = [0 1; 1 0];
Y = [0 -1i; 1i 0];
Z = [1 0; 0 -1];
gam = 0:T/sample:T;

JR = zeros(4,4,sample); 
JC = zeros(4,4,sample); 

for j = 1:sample
    JR(:,:,j) = AmplitudeDamping(gam(j)) / d;
    JC(:,:,j) = AmplitudeDamping(gam(j));
end

total_dim = 2^4;

%% primal
cvx_begin sdp
    variable J1(total_dim, total_dim) hermitian % 1234
    variable J2(total_dim, total_dim) hermitian
    % variable p1
    % variable p2

    p1 = 2;
    p2 = 1;
    
    J = J1 - J2;
    
    cost = p1 + p2;
    minimize cost
    
    for j = 1:sample
        JUo = PartialTrace(J*Tensor(eye(d), JR(:,:,j).', eye(d)), 2, [d, 4, d]);
        JUo == JC(:,:,j);
    end
    
    PartialTrace(J1, 3, [d, 4, d]) == p1*eye(d * 4);
    PartialTrace(J2, 3, [d, 4, d]) == p2*eye(d * 4);
    
    J1 >= 0; J2 >= 0;
    
cvx_end
    
cost

% functions
function J_ad = AmplitudeDamping(gam)
    E0 = [1 0; 0 sqrt(1-gam)];
    E1 = [0 sqrt(gam); 0 0];
    MES = 2*MaxEntangled(2,0,1) * MaxEntangled(2,0,1)';
    J_ad = kron(eye(2), E0) * MES * kron(eye(2), E0)' + kron(eye(2), E1) * MES * kron(eye(2), E1)'; 
end
