clear; clc;

%% model setup

d = 2;

sample = 1000;

n = 1;

T = 10;

I = eye(d);

X = [0 1; 1 0];

Y = [0 -1i; 1i 0];

Z = [1 0; 0 -1];

% Lindbladian

% amplitude damping

G = 0.1;

L0 = sqrt(G)*[0 1; 0 0];

L_ops = {L0};

H = 0 * Z;

%% Choi state resource

JC = zeros(d^2,d^2,sample);

JUn = zeros(d^(2*n),d^(2*n),sample); % Caution: Here we store Choi state!

t = 0:T/sample:T;

for j = 1:sample

JC(:, :, j) = LindbladtoChoi(d, t(j), H, L_ops) / d;

temp = 1;

for k = 1:n

temp = kron(temp, JC(:, :, j));

end

JUn(:, :, j) = temp;

end

total_dim = d^2 * d^(2*n);


err = 0:0.2/40:0.2;

cost_array = zeros(1, numel(err));

for l=1:numel(err)

%% primal

cvx_begin sdp quiet

variable J1(total_dim, total_dim) hermitian % 1234

variable J2(total_dim, total_dim) hermitian

variable p1

variable p2

J = J1 - J2;

cost = p1 + p2;

minimize cost

for j = 1:sample

JUo = PartialTrace(J*Tensor(eye(d), JUn(:,:,j).', eye(d)), 2, [d, d^(2*n), d]);

0.5*DiamondNorm(JUo - d*JC(:,:,j)) <= err(l);

end

PartialTrace(J1, 3, [d, d^(2*n), d]) == p1*eye(d^(2*n+1));

PartialTrace(J2, 3, [d, d^(2*n), d]) == p2*eye(d^(2*n+1));

PartialTrace(J, 3, [d, d^(2*n), d]) == eye(d^(2*n+1));

J1 >= 0; J2 >= 0;

cvx_end

cost_array(l) = cost

end