clear; clc;


%% model setup 
d = 4;
sample = 1000;
n = 1;
T = 4*pi;
I = eye(d);
X = [0 1; 1 0];
Y = [0 -1i; 1i 0];
Z = [1 0; 0 -1];

P0 = [1 0; 0 0];
P1 = [0 0; 0 1];
v0 = [1;0];
v1 = [0;1];
t = 0:T/sample:T;

b00 = [1; 0; 0; 0];
b11 = [0; 0; 0; 1];
bPhi = (1/sqrt(2))*[1; 0; 0; 1];
bPsi = (1/sqrt(2))*[0; 1; -1; 0];

%
lam = 1;
L0 = lam*b00*b00';
L1 = lam*b11*b11';
L2 = lam*bPhi*bPhi';
L3 = lam*bPsi*bPsi';
L_ops = {0*L0};
H = [-1 0 0 0; 0 0 -1 0; 0 -1 0 0; 0 0 0 -1];


%% one parameter state resource
JR = zeros(4,4,sample);
JC = zeros(d^2,d^2,sample);
JRn = zeros(4^(n),4^(n),sample); 
for j = 1:sample
    psi = (1/2) * (exp(-1i*2*t(j)) * bPsi + b00 + b11 + bPhi);
    JR(:, :, j) = psi*psi'/trace(psi*psi');
    JC(:, :, j) = LindbladtoChoi(d, t(j), H, L_ops);
    temp = 1;
    for k = 1:n
        temp = kron(temp, JR(:, :, j));
    end
    JRn(:, :, j) = temp;
end

total_dim = d^2 * 4^(n);

%% primal
cvx_begin sdp
    variable J1(total_dim, total_dim) hermitian % 1234
    variable J2(total_dim, total_dim) hermitian
    variable p1
    variable p2

    J = J1 - J2;

    cost = p1 + p2;
    minimize cost

    for j = 1:sample
        JUo = PartialTrace(J*Tensor(eye(d), JRn(:,:,j).', eye(d)), 2, [d, 4^(n), d]);
        JUo == JC(:,:,j);
    end

    PartialTrace(J1, 3, [d, 4^(n), d]) == p1*eye(d * 4^n);
    PartialTrace(J2, 3, [d, 4^(n), d]) == p2*eye(d * 4^n);

    J1 >= 0; J2 >= 0;

cvx_end

primal_cost = cost
