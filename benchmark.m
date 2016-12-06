N = 1000;
sig_X = 10;
sig_eps = 1;
W_true = [10, -20, 50, -20, 10];
d = size(W_true, 1);

X = normrnd(zeros(N, d), sig_X);
y = X*W_true + normrnd(zeros(N, 1), sig_X);