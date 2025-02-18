clc; 
clear; 

% Given Prior Probabilities
P_w1 = 0.2;  % Prior probability of cancer patients
P_w2 = 0.8;  % Prior probability of non-cancer patients

% Given Loss Matrix
L = [0  10;   % Risk of choosing chemotherapy
     20  0];  % Risk of choosing medication

% Compute Risks
R_a1 = L(1,1) * P_w1 + L(1,2) * P_w2;  % Risk for chemotherapy
R_a2 = L(2,1) * P_w1 + L(2,2) * P_w2;  % Risk for medication

% Decision based on minimum risk
if R_a1 < R_a2
    decision = 'Choose Chemotherapy (a1)';
else
    decision = 'Choose Medication (a2)';
end

% Display Results
fprintf('Prior Probability of Cancer Patients (P(w1)): %.2f\n', P_w1);
fprintf('Prior Probability of Non-Cancer Patients (P(w2)): %.2f\n', P_w2);
fprintf('Risk for Chemotherapy (a1): %.2f\n', R_a1);
fprintf('Risk for Medication (a2): %.2f\n', R_a2);
fprintf('Optimal Decision: %s\n', decision);
