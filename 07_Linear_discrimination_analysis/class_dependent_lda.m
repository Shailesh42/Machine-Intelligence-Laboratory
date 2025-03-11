clc; clear; close all;
num_classes = 3; 
num_samples = 30;
num_features = 2;
rng(42); 

% Mean values for each class
mu = [2 2; 6 6; 10 2]; 
sigma = eye(num_features); 

% Generate data
X = [];
y = [];
for c = 1:num_classes
    X_class = mvnrnd(mu(c, :), sigma, num_samples);
    X = [X; X_class];
    y = [y; c * ones(num_samples, 1)];
end

% Compute mean for each class
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

% Compute within-class scatter matrices for each class
Sw_class = cell(num_classes, 1);
W_class = cell(num_classes, 1);
X_lda_class = cell(num_classes, 1);

for c = 1:num_classes
    X_c = X(y == c, :);
    Sw_class{c} = (X_c - class_means(c, :))' * (X_c - class_means(c, :));
    
    % Solve generalized eigenvalue problem for class-specific LDA
    [V, D] = eig(Sb, Sw_class{c});
    [~, sorted_indices] = sort(diag(D), 'descend'); 
    W_class{c} = V(:, sorted_indices(1)); 

    % Project data onto class-specific projection line
    X_lda_class{c} = X_c * W_class{c}; 
end

% Plot Original 2D Data
figure;
subplot(1,2,1);
gscatter(X(:,1), X(:,2), y, ['r', 'g', 'b'], ['o', 's', '^'], 5);
title('Original 2D Data');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');
axis square; 

% Plot Class-Dependent LDA Projection
subplot(1,2,2); hold on;
colors = ['r', 'g', 'b'];
markers = {'o', 's', '^'};

% Plot separate projection lines for each class
for c = 1:num_classes
    scatter(X_lda_class{c}, c * ones(size(X_lda_class{c})), 50, colors(c), markers{c});
end

title('Class-Dependent LDA Projection');
xlabel('LDA Component');
ylabel('Projection Line');
legend('Class 1', 'Class 2', 'Class 3');
yticks(1:num_classes);
yticklabels({'Class 1', 'Class 2', 'Class 3'});
axis square;
hold off;
