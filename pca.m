load fisheriris
X = meas;
labels = species;

X_mean = mean(X);
X_std = std(X);
X_norm = (X - X_mean) ./ X_std;

cov_matrix = cov(X_norm);

[eig_vectors, eig_values_matrix] = eig(cov_matrix);
eig_values = diag(eig_values_matrix);

[eig_values_sorted, idx] = sort(eig_values, 'descend');
eig_vectors_sorted = eig_vectors(:, idx);

W = eig_vectors_sorted(:, 1:2);
Z = X_norm * W;

gscatter(Z(:,1), Z(:,2), labels)
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
title('PCA: Iris Dataset (Manual)')
grid on