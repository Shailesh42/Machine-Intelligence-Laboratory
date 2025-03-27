   clc; clear; close all;
X = [2 1; 2 -1; 4 0; 1 2; 3 1; 3 -2; 2 0; 4 1; 5 -2;
     5 0; 6 2; 7 -1; 6 -2; 7 1; 8 0; 5 3; 6 -3; 7 -2];  

Y = [1; 1; 1; 1; 1; 1; 1; 1; 1;
    -1; -1; -1; -1; -1; -1; -1; -1; -1];
[m, n] = size(X);
H = (Y * Y') .* (X * X');  
f = -ones(m, 1);
Aeq = Y';
beq = 0;
lb = zeros(m, 1);
alpha = quadprog(H, f, [], [], Aeq, beq, lb, []);
sv_idx = find(alpha > 1e-5);
support_alpha = alpha(sv_idx);
support_Y = Y(sv_idx);
support_X = X(sv_idx, :);

w = sum((support_alpha .* support_Y) .* support_X, 1)';

b = mean(support_Y - support_X * w);

fprintf('Optimal Weight Vector:\n'); disp(w);
fprintf('Optimal Bias:\n'); disp(b);

figure; hold on;
gscatter(X(:,1), X(:,2), Y, 'rb', 'ox', 10);
xlabel('Feature 1'); ylabel('Feature 2');
title('SVM with Optimal Hyperplane and Support Vectors');

[x1Grid, x2Grid] = meshgrid(linspace(min(X(:,1))-1, max(X(:,1))+1, 100), ...
                            linspace(min(X(:,2))-1, max(X(:,2))+1, 100));
gridX = [x1Grid(:), x2Grid(:)];

decisionValues = gridX * w + b;
decisionValues = reshape(decisionValues, size(x1Grid));

contour(x1Grid, x2Grid, decisionValues, [0 0], 'k', 'LineWidth', 2);

scatter(support_X(:,1), support_X(:,2), 100, 'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

marginValues = (gridX * w + b); 
marginValues = reshape(marginValues, size(x1Grid));

contour(x1Grid, x2Grid, marginValues, [-1 -1], '--b', 'LineWidth', 1.5);
contour(x1Grid, x2Grid, marginValues, [1 1], '--r', 'LineWidth', 1.5);

legend('Class 1', 'Class 2', 'Decision Boundary', 'Support Vectors', '+1 Margin', '-1 Margin');
hold off;
