clc; clear; close all;

X1 = [2, 2; 2, -2; -2, -2; -2, 2];
X2 = [1, 1; 1, -1; -1, 1; -1, -1];
Y1 = ones(size(X1, 1), 1); 
Y2 = -ones(size(X2, 1), 1); 
X = [X1; X2];
Y = [Y1; Y2];

figure; hold on;
scatter(X1(:,1), X1(:,2), 100, 'r', 'o', 'filled');  
scatter(X2(:,1), X2(:,2), 100, 'b', 's', 'filled');  
xlabel('X1'); ylabel('X2'); title('Original Data (Non-Linearly Separable)');
legend('Class +1', 'Class -1'); grid on;
hold off;

gamma = 0.5;
Z = rbfKernelTransform(X, gamma);
SVMModel = fitcsvm(Z, Y, 'KernelFunction', 'linear', 'Standardize', true);

plotDecisionBoundary(SVMModel, Z, Y, 'Linearly Separable Form (RBF Kernel)');
plotNonLinearBoundary(SVMModel, X, Y, gamma);

function Z = rbfKernelTransform(X, gamma)
    Z = zeros(size(X, 1), 2);
    for i = 1:size(X, 1)
        x1 = X(i, 1);
        x2 = X(i, 2);
        Z(i, :) = [exp(-gamma * (x1^2 + x2^2)), exp(-gamma * ((x1 - x2)^2))];
    end
end

function plotDecisionBoundary(SVMModel, X, Y, titleStr)
    figure; hold on;
    gscatter(X(:,1), X(:,2), Y, 'rb', 'os', 10, 'filled');
    [x1Grid, x2Grid] = meshgrid(linspace(min(X(:,1))-1, max(X(:,1))+1, 100), linspace(min(X(:,2))-1, max(X(:,2))+1, 100));
    XGrid = [x1Grid(:), x2Grid(:)];
    [~, score] = predict(SVMModel, XGrid);
    contour(x1Grid, x2Grid, reshape(score(:,2), size(x1Grid)), [0, 0], 'k', 'LineWidth', 2);
    sv = SVMModel.SupportVectors;
    plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('Feature 1'); ylabel('Feature 2'); title(titleStr);
    legend('Class +1', 'Class -1', 'Decision Boundary', 'Support Vectors');
    grid on;
    hold off;
end

function plotNonLinearBoundary(SVMModel, X, Y, gamma)
    figure; hold on;
    gscatter(X(:,1), X(:,2), Y, 'rb', 'os', 10, 'filled');
    [x1Grid, x2Grid] = meshgrid(linspace(min(X(:,1))-1, max(X(:,1))+1, 100), linspace(min(X(:,2))-1, max(X(:,2))+1, 100));
    XGrid = [x1Grid(:), x2Grid(:)];
    ZGrid = rbfKernelTransform(XGrid, gamma);
    [~, score] = predict(SVMModel, ZGrid);
    contour(x1Grid, x2Grid, reshape(score(:,2), size(x1Grid)), [0, 0], 'k', 'LineWidth', 2);
    sv_original = X(SVMModel.IsSupportVector, :);
    plot(sv_original(:,1), sv_original(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('X1'); ylabel('X2'); title('Non-Linear Decision Boundary in Original Space (RBF Kernel)');
    legend('Class +1', 'Class -1', 'Decision Boundary', 'Support Vectors');
    grid on;
    hold off;
end
