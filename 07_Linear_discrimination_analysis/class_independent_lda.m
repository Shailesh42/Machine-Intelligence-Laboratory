clc; clear; close all;

num_classes = 3; 
num_samples = 30;
num_features = 2;
rng(42);

mu = [2 2; 6 6; 10 2]; 
sigma = eye(num_features);

X = [];
y = [];
for c = 1:num_classes
    X_class = mvnrnd(mu(c, :), sigma, num_samples);
    X = [X; X_class];
    y = [y; c * ones(num_samples, 1)];
end

class_means = zeros(num_classes, num_features);
for c = 1:num_classes
    class_means(c, :) = mean(X(y == c, :), 1);
end
overall_mean = mean(X, 1);

Sw = zeros(num_features, num_features);
Sb = zeros(num_features, num_features);
for c = 1:num_classes
    X_c = X(y == c, :);
    Nc = sum(y == c);
    
    mean_diff = (class_means(c, :) - overall_mean)'; 
    Sb = Sb + Nc * (mean_diff * mean_diff');  % Between-class scatter
    
    Sw = Sw + (X_c - class_means(c, :))' * (X_c - class_means(c, :)); % Within-class scatter
end

[V, D] = eig(inv(Sw) * Sb);
[~, sorted_indices] = sort(diag(D), 'descend'); 
W = V(:, sorted_indices(1));

X_lda = X * W;

figure;
subplot(1,2,1);
gscatter(X(:,1), X(:,2), y, ['r', 'g', 'b'], ['o', 's', '^'], 5);
title('Original 2D Data');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');
axis square; 
hold on;
quiver(overall_mean(1), overall_mean(2), W(1), W(2), 3, 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5);
hold off;

subplot(1,2,2);
hold on;
markers = {'o', 's', '^'};
colors = ['r', 'g', 'b'];
for c = 1:num_classes
    scatter(X_lda(y == c), zeros(sum(y == c), 1), 50, colors(c), markers{c});
end
title('LDA Projection (1D)');
xlabel('Projected Feature');
ylabel('Class Separation');
legend('Class 1', 'Class 2', 'Class 3');
axis tight;
hold off;
