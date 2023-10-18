n_feat =  78*77/2;
n_t = 30000;

a = randn(n_feat, n_t);
b = randn(n_feat, n_t);

tic
for i = 1:n_feat
    corr(a(i, :)', b(i, :)');
end
disp(toc)

tic
fast_corr(a', b');
disp(toc)