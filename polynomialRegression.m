clc; clear;

%%Example -1
load carsmall
validIdx = ~isnan(Horsepower) & ~isnan(MPG);

xData = Horsepower(validIdx);
yData = MPG(validIdx);

degree = 3;

% Fit polynomial curve
coefficients = fitPolynomial(xData, yData, degree);

% Print the polynomial equation
printPolynomial(coefficients);

newX = 130;
predictedY = predict(coefficients, newX);
fprintf('\nFor X = %d, predicted Y value is: %.2f\n', newX, predictedY);

yDataPredicted = predict(coefficients, xData);
err=[yData;yDataPredicted;abs(yData-yDataPredicted);abs((yData-yDataPredicted)./yData*100)]';
meanError=mean(err);

newx=[min(xData) : 0.1 : max(xData)];
newy = predict(coefficients, newx);

hold all;
plot(xData, yData, 'o');
plot(newx, newy, '-');
xlabel('Horsepower');
ylabel('MPG');
title('3rd order polynomial regression manually');
grid on;


function coefficients = fitPolynomial(xData, yData, degree)
    n = length(xData);
    matrix = zeros(degree + 1, degree + 2);

    for i = 1:degree + 1
        for j = 1:degree + 1
            matrix(i, j) = sum(xData.^(i + j - 2));
        end
        matrix(i, degree + 2) = sum(yData .* xData.^(i - 1));
    end

    coefficients = solveEquations(matrix);
end

function coefficients = solveEquations(matrix)
    n = size(matrix, 1);
    coefficients = zeros(n, 1);

    for i = 1:n
        for j = i + 1:n
            factor = matrix(j, i) / matrix(i, i);
            matrix(j, :) = matrix(j, :) - factor * matrix(i, :);
        end
    end

    for i = n:-1:1
        coefficients(i) = matrix(i, n + 1) / matrix(i, i);
        for j = i - 1:-1:1
            matrix(j, n + 1) = matrix(j, n + 1) - matrix(j, i) * coefficients(i);
            matrix(j, i) = 0;
        end
    end
end

function printPolynomial(coefficients)
    fprintf('coefficients:\n');
    disp(coefficients);
    fprintf('Generated Polynomial Equation: y = ');
    for i = length(coefficients):-1:1
        if i == 1
            fprintf('%.2f', coefficients(i));
        else
            fprintf('%.2fx^%d + ', coefficients(i), i - 1);
        end
    end
    fprintf('\n');
end

function predictedY = predict(coefficients, x)
    predictedY = zeros(size(x));
    for i = 1:length(coefficients)
        predictedY = predictedY + coefficients(i) .* x.^(i-1);
    end
end



%%




%%Example -2

clc; clear;

load carsmall
validIdx = ~isnan(Horsepower) & ~isnan(MPG);
xData = Horsepower(validIdx);
yData = MPG(validIdx);

degrees = [1, 10, 20];

xData = zscore(xData);

xFit = linspace(min(xData), max(xData), 500);
colors = lines(length(degrees));
figure;
hold on;

legendEntries = cell(1, length(degrees));
for i = 1:length(degrees)
    degree = degrees(i);

    coefficients = polyfit(xData, yData, degree);

    % You might need to define a printPolynomial function
    % printPolynomial(coefficients); % coefficients(1), coefficients(2)...

    yPred = polyval(coefficients, xData);
    yFit = polyval(coefficients, xFit);

    SS_res = sum((yData - yPred).^2);
    SS_tot = sum((yData - mean(yData)).^2);
    R_squared = 1 - (SS_res / SS_tot);
    MAPE = mean(abs((yData - yPred) ./ yData)) * 100;

    plot(xFit, yFit, 'LineWidth', 1.8, 'Color', colors(i,:));

    legendEntries{i} = sprintf('Degree %d | R^2=%.4f | MAPE=%.2f%%', degree, R_squared, MAPE);
end

plot(xData, yData, 'ko', 'MarkerEdgeColor', 'k');

xlabel('Horsepower');
ylabel('MPG');
title('Polynomial Regression for Various Degrees');
legend(legendEntries, 'location', 'northeast');
grid on;

% function printPolynomial(degree, coefficients)
% 
%     fprintf('Degree %d Polynomial Equation:\ny = ', degree);
% 
%     for j = 1:length(coefficients)
%         power = length(coefficients) - j;
%         coeff = coefficients(j);
%         if coeff >= 0 && j > 1
%             fprintf('+');
%         end
%         if power > 0
%             fprintf('%.2fx^%d ', coeff, power);
%         else
%             fprintf('%.2f ', coeff);
%         end
%     end
%     fprintf('\n');
% end





%%



%%Example -3

clc; clear;

load carsmall
validIdx = ~isnan(Horsepower) & ~isnan(MPG);
xData = Horsepower(validIdx);
yData = MPG(validIdx);

xData = zscore(xData);

degrees = 1:33;
r2_scores = zeros(size(degrees));
mapes = zeros(size(degrees));

for i = 1:length(degrees)
    degree = degrees(i);

    coefficients = polyfit(xData, yData, degree);
    yPred = polyval(coefficients, xData);

    r2_scores(i) = 1 - sum((yData - yPred).^2) / sum((yData - mean(yData)).^2);
    mapes(i) = mean(abs((yData - yPred) ./ yData)) * 100;
end

score = r2_scores - mapes;
[~, bestIdx] = max(score);

fprintf('Optimal degree: %d\n', degrees(bestIdx));
fprintf('R^2 = %.4f\n', r2_scores(bestIdx));
fprintf('MAPE = %.2f%%\n', mapes(bestIdx));

figure;
yyaxis left
plot(degrees, r2_scores, '-o', 'LineWidth', 2);
ylabel('R^2');
ylim([0 1]);
ax = gca;
ax.YColor = 'b';

yyaxis right
plot(degrees, mapes, '--s', 'LineWidth', 2);
ylabel('MAPE (%)');
ax.YColor = 'r';

xlabel('Polynomial Degree');
title('Polynomial Degree vs R^2 and MAPE');
legend('R^2','MAPE (%)');
grid on;