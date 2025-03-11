clc; clear; close all;

% Parameters
num_classes = 3;
num_samples = 30;
num_features = 2;
rng(42);

% Mean and covariance
mu = [2 2; 6 6; 10 2]; 
sigma = eye(num_features);

% Data storage
X = [];
y = [];

% Generate data for each class
for c = 1:num_classes
    X_class = mvnrnd(mu(c, :), sigma, num_samples);
    X = [X; X_class];
    y = [y; c * ones(num_samples, 1)];
end

% Compute class means
class_means = zeros(num_classes, num_features);
for c = 1:num_classes
    class_means(c, :) = mean(X(y == c, :), 1);
end
overall_mean = mean(X, 1);

% Compute between-class scatter matrix Sb
Sb = zeros(num_features, num_features);
for c = 1:num_classes
    Nc = sum(y == c);
    mean_diff = (class_means(c, :) - overall_mean)';
    Sb = Sb + Nc * (mean_diff * mean_diff');
end

% Compute within-class scatter matrix Sw
Sw = zeros(num_features, num_features);
for c = 1:num_classes
    X_c = X(y == c, :);
    Sw = Sw + (X_c - class_means(c, :))' * (X_c - class_means(c, :));
end

% Solve eigenvalue problem
[V, D] = eig(Sb, Sw);
[~, sorted_indices] = sort(diag(D), 'descend');
W = V(:, sorted_indices(1));
X_lda = X * W;

% Plot results
figure;

% Original 2D Data
subplot(1,2,1);
gscatter(X(:,1), X(:,2), y, ['r', 'g', 'b'], ['o', 's', '^'], 5);
title('Original 2D Data');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');
axis square;

% Class-Independent LDA Projection
subplot(1,2,2);
hold on;
unique_classes = unique(y);
markers = {'o', 's', '^'}; 
colors = {'r', 'g', 'b'}; 

for i = 1:length(unique_classes)
    class_idx = y == unique_classes(i);
    scatter(X_lda(class_idx), zeros(sum(class_idx),1), 50, colors{i}, markers{i});
end
plot([min(X_lda), max(X_lda)], [0, 0], '--k'); % Projection axis
hold off;
title('Class-Independent LDA Projection');
xlabel('LDA Component');
ylabel('Projection Line');
legend('Class 1', 'Class 2', 'Class 3');
axis square;
