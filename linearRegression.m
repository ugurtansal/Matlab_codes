data=readtable("auto_mpg.csv");

summary(data);
data=rmmissing(data) %%Yalnızca 2 değerde NaN olmasına rağmen hiç bir sonuç çıkmadı
                       %%O yüzden verileri temizlemek önemli

X=[data.displacement,data.horsepower,data.weight,...
    data.cylinders,data.acceleration,data.model_year];

y=data.mpg;

cv=cvpartition(size(X,1),"HoldOut",0.2);
X_train=X(training(cv),:);
y_train=y(training(cv),:);
X_test=X(test(cv),:);
y_test=y(test(cv),:);

fprintf("Train data: %d rows \nTest data: %d rows \n\n",...
    size(X_train,1),size(X_test,1));


beta_backslash= X_train \ y_train;

beta_inv= inv(X_train' * X_train) * (X_train' * y_train);

beta_pinv=pinv(X_train) * y_train;

y_pred_backslash=X_test * beta_backslash;
y_pred_inv= X_test * beta_inv;
y_pred_pinv=X_test*beta_pinv;


function evaluate(y_true,y_pred)
    mse=mean((y_true-y_pred).^2);
    rmse=sqrt(mse);
    mae=mean(abs(y_true-y_pred));
    mape=mean(abs((y_true-y_pred)./y_true))*100;
    r2=1-sum((y_true-y_pred).^2)/sum((y_true-mean(y_true)).^2);

    fprintf('MSE:%.2f | RMSE: %.2f | MAE: %.2f | MAPE:%.2f | R^2:%.4f \n\n',...
        mse,rmse,mae,mape,r2)
end


disp('Backslash method: '); evaluate(y_test,y_pred_backslash);
disp('inv() method: '); evaluate(y_test,y_pred_inv);
disp('pinv() method: '); evaluate(y_test,y_pred_pinv);




%%




%%Lineer Regression - 2

data=readtable("auto_mpg.csv",'VariableNamingRule','preserve');

data=rmmissing(data) 

X=[data.displacement,data.horsepower,data.weight,...
    data.cylinders,data.acceleration,data.("model-year")];

y=data.mpg;

cv=cvpartition(size(X,1),"HoldOut",0.2);
X_train=X(training(cv),:);
y_train=y(training(cv),:);
X_test=X(test(cv),:);
y_test=y(test(cv),:);

fprintf("Train data: %d rows \nTest data: %d rows \n\n",...
    size(X_train,1),size(X_test,1));


%Method 1 : Using regress
fprintf("--------------Using regress ----------------")
[beta_regress,bint,r,rint,stats]=regress(y_train,[ones(size(X_train,1),1) X_train]);
p_values_regress=stats(3);

%Add intercept term to X_test for prediction 
X_test_with_intercept=[ones(size(X_test,1),1) X_test];
y_pred_regress=X_test_with_intercept * beta_regress;

disp("Regression coefficients with regress: ");
disp(beta_regress);

fprintf("Model p-value: %.4f\n\n",p_values_regress);

%Method 2: Using fitlm (more comprehensive)
fprintf('-------------Using fitlm ---------------');

tbl=array2table([X_train y_train],'VariableNames',...
    {'displacement','horsepower','weight','cylinders','acceleration','model_year','mpg'});

model=fitlm(tbl,'mpg ~ displacement + horsepower + weight + cylinders + acceleration + model_year');

disp(model);

%Predict on test data
tbl_test=array2table(X_test,'VariableNames',...
    {'displacement','horsepower','weight','cylinders','acceleration','model_year'});
y_pred_fitlm=predict(model,tbl_test);

%Evaluate both methods
disp(' ');
disp('regress method: ');evaluate(y_test,y_pred_regress);
disp('fitlm method: ');evaluate(y_test,y_pred_fitlm)
% function evaluate(y_true,y_pred)
%     mse=mean((y_true-y_pred).^2);
%     rmse=sqrt(mse);
%     mae=mean(abs(y_true-y_pred));
%     mape=mean(abs((y_true-y_pred)./y_true))*100;
%     r2=1-sum((y_true-y_pred).^2)/sum((y_true-mean(y_true)).^2);
% 
%     fprintf('MSE:%.2f | RMSE: %.2f | MAE: %.2f | MAPE:%.2f | R^2:%.4f \n\n',...
%         mse,rmse,mae,mape,r2)
% end

%%



%%Lineer Regresyon 3

clc; clear;

data = readtable('auto_mpg.csv', 'VariableNamingRule', 'preserve');
data = rmmissing(data);

X = [data.displacement, data.horsepower, data.weight, ...
     data.cylinders, data.acceleration, data.("model-year")];
y = data.mpg;

cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

fprintf("Train data: %d rows \nTest data: %d rows\n\n", ...
    size(X_train,1), size(X_test,1));

model = multipleLinearRegression(X_train, y_train);

y_pred = model.intercept + X_test * model.weights;

disp('Test Performance:');
evaluate(y_test, y_pred);

disp('Actual and Predicted mpg (first 5 examples):');
disp(table(y_test(1:5), y_pred(1:5), 'VariableNames', {'Actual_MPG','Predicted_MPG'}));
disp(model.lm);

function model = multipleLinearRegression(X, y)
    T = array2table(X, 'VariableNames', {'displacement','horsepower',...
        'weight','cylinders','acceleration','model_year'});
    T.mpg = y;

    lm = fitlm(T, 'mpg ~ displacement + horsepower + weight + cylinders + acceleration + model_year');

    model.intercept = lm.Coefficients.Estimate(1);     % beta_0 (intercept)
    model.weights = lm.Coefficients.Estimate(2:end);   % beta_1, ..., beta_n
    model.lm = lm;
end

function evaluate(y_true, y_pred)
    mse = mean((y_true - y_pred).^2);
    rmse = sqrt(mse);
    mae = mean(abs(y_true - y_pred));
    mape = mean(abs((y_true - y_pred)./y_true)) * 100;
    r2 = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);

    fprintf('MSE: %.2f | RMSE: %.2f | MAE: %.2f | MAPE: %.2f%% | R^2: %.4f\n\n', ...
        mse, rmse, mae, mape, r2);
end

function model = multipleLinearRegression2(X, y)
    X = [ones(size(X, 1), 1), X];
    w = pinv(X' * X) * X' * y;
    model.intercept = w(1);
    model.weights = w(2:end);
end
