clear; clc;


%% model setup 
d = 2;
sample = 4000;
n = 1;
T = 4*pi;
I = eye(d);
X = [0 1; 1 0];
Y = [0 -1i; 1i 0];
Z = [1 0; 0 -1];
t = 0:T/sample:T;

%% one parameter state resource
n = 1;
P0 = [1 0; 0 0];
P1 = [0 0; 0 1];
v0 = [1;0];
v1 = [0;1];

JR = zeros(2,2,sample);
JC = zeros(d^2,d^2,sample);
JRn = zeros(2^(n),2^(n),sample); 
for j = 1:sample
    % psi = (1/sqrt(2))*(kron(v0, v0) + exp(1i*2*t(j))*kron(v1, v1));
    psi = (1/sqrt(2))*(v0 + exp(1i*2*t(j))*v1);

    JR(:, :, j) = psi * psi';
    JC(:, :, j) = ChoiMatrix({expm(1i*t(j)*Z)});
    temp = 1;
    for k = 1:n
        temp = kron(temp, JR(:, :, j));
    end
    JRn(:, :, j) = temp;
end

total_dim = d^2 * 2^n;

% primal
cvx_begin sdp
    variable J1(total_dim, total_dim) hermitian % 1234
    variable J2(total_dim, total_dim) hermitian
    % variable p1
    % variable p2

    J = J1 - J2;

    p1 = 2;
    p2 = 1;

    cost = p1 + p2;
    minimize cost

    for j = 1:sample
        JUo = PartialTrace(J*Tensor(eye(d), JRn(:,:,j).', eye(d)), 2, [d, 2^(n), d]);
        JUo == JC(:,:,j);
    end

    PartialTrace(J1, 2, [2*(2^n), 2]) == p1*eye(2*(2^n));
    PartialTrace(J2, 2, [2*(2^n), 2]) == p2*eye(2*(2^n));

    J1 >= 0; J2 >= 0;
    

cvx_end

primal_cost = cost