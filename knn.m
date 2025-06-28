data = readtable('wine.csv', 'FileType', 'text', 'ReadVariableNames', false);

data.Properties.VariableNames = {'Class', 'Alcohol', 'MalicAcid', 'Ash', ...
    'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids', ...
    'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', ...
    'Hue', 'OD280_OD315', 'Proline'};

X = data{:, 2:end};
Y = data{:, 1};

X = zscore(X);

cv = cvpartition(Y, 'HoldOut', 0.3);
XTrain = X(training(cv), :);
YTrain = Y(training(cv));
XTest = X(test(cv), :);
YTest = Y(test(cv));

YPred = manuelKNN(XTrain, YTrain, XTest, 5);

confMat = confusionmat(YTest, YPred);
disp('Confusion Matrix:');
disp(confMat);
confusionchart(YTest, YPred);
title('KNN Confusion Matrix - Wine Dataset');

accuracy = sum(diag(confMat)) / sum(confMat(:));
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

function Ypred = manuelKNN(Xtrain, Ytrain, Xtest, K)
    numTest = size(Xtest, 1);
    Ypred = zeros(numTest, 1);
    for i = 1:numTest
        x = Xtest(i, :);
        dists = sqrt(sum((Xtrain - x).^2, 2));
        [~, idx] = sort(dists);
        nearestLabels = Ytrain(idx(1:K));
        Ypred(i) = mode(nearestLabels);
    end
end


%%




%%Example -2
data = readtable('BostonHousing.csv');
X = data{:, 1:end-1};
Y = data{:, end};

X(:, std(X) == 0) = [];  % Sabit sütunları kaldır
X = zscore(X);           % Normalize et

cv = cvpartition(size(X,1), 'HoldOut', 0.3);
XTrain = X(training(cv), :);
YTrain = Y(training(cv));
XTest = X(test(cv), :);
YTest = Y(test(cv));

YPred = manuelKNNReg(XTrain, YTrain, XTest, 5);

mse = mean((YTest - YPred).^2);
mae = mean(abs(YTest - YPred));
r2 = 1 - sum((YTest - YPred).^2) / sum((YTest - mean(YTest)).^2);

fprintf('Mean Squared Error (MSE): %.2f\n', mse);
fprintf('Mean Absolute Error (MAE): %.2f\n', mae);
fprintf('R^2 Score: %.2f\n', r2);

function Ypred = manuelKNNReg(Xtrain, Ytrain, Xtest, K)
    numTest = size(Xtest, 1);
    Ypred = zeros(numTest, 1);

    for i = 1:numTest
        x = Xtest(i, :);
        dists = sqrt(sum((Xtrain - x).^2, 2));
        [~, idx] = sort(dists);
        nearestValues = Ytrain(idx(1:K));
        Ypred(i) = mean(nearestValues);
    end
end
