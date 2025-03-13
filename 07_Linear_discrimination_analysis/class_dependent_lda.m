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

Sb = zeros(num_features, num_features);
Sw_class = cell(num_classes, 1);
W_class = cell(num_classes, 1);
X_lda_class = cell(num_classes, 1);

for c = 1:num_classes
    X_c = X(y == c, :);
    Nc = sum(y == c);
    
    mean_diff = (class_means(c, :) - overall_mean)';
    Sb = Sb + Nc * (mean_diff * mean_diff'); 

    Sw_class{c} = (X_c - class_means(c, :))' * (X_c - class_means(c, :));
    [V, D] = eig(inv(Sw_class{c}) * Sb);
    [~, sorted_indices] = sort(diag(D), 'descend'); 
    W_class{c} = V(:, sorted_indices(1));  
    X_lda_class{c} = (X_c - class_means(c, :)) * W_class{c};  
end

figure;
subplot(1,2,1);
gscatter(X(:,1), X(:,2), y, ['r', 'g', 'b'], ['o', 's', '^'], 5);
title('Original 2D Data');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');
axis square; 
hold on;

colors = ['r', 'g', 'b'];
for c = 1:num_classes
    mean_point = class_means(c, :);  
    quiver(mean_point(1), mean_point(2), W_class{c}(1), W_class{c}(2), 2, colors(c), 'LineWidth', 2, 'MaxHeadSize', 0.5);
end
hold off;

subplot(1,2,2);
hold on;
markers = {'o', 's', '^'};

for c = 1:num_classes
    projected_data = class_means(c, :) + X_lda_class{c} * W_class{c}';
    scatter(projected_data(:,1), projected_data(:,2), 50, colors(c), markers{c});
end

title('Class-Dependent LDA Projection');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');
axis square;
hold off;
