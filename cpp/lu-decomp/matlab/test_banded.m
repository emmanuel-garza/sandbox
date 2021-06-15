n = 100000;

n_band_left = 20;
n_band_right = 20;

A = sparse(n, n);

% c = 1;
% for ind = 1:n
%     for ind2 = max(1, ind-n_band_left) : min(n, ind+n_band_right)
%         A(ind, ind2) = 1.1;
%         c = c+1;
%     end
% end


e = ones(n,1);
A = spdiags(1.1*repmat(e,1,n_band_left+n_band_right+1), -n_band_left:n_band_right, n, n );



%return

% A = A + sparse(diag(

b = ones(n,1);

%[L, U] = lu(A');

y = A \ b;

%y = U \ (L \ b);

y(1:10)


condest(A)