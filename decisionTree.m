clear;clc;
Outlook = {'Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', ...
           'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'}';

Humidity = {'High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', ...
            'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'}';

Wind = {'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', ...
        'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'}';

PlayTennis = {'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', ...
              'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'}';

data = table(Outlook, Humidity, Wind, PlayTennis);
tree = fitctree(data, 'PlayTennis');
view(tree, 'Mode', 'graph'); % Ağaç grafiğini göster

data = table(Outlook, Humidity, Wind, PlayTennis);
tree = fitctree(data, 'PlayTennis');

testData = table({'Sunny'; 'Overcast'; 'Rain'}, ...
                  {'High'; 'Normal'; 'Normal'}, ...
                  {'Strong'; 'Weak'; 'Strong'}, ...
                  'VariableNames', {'Outlook', 'Humidity', 'Wind'});

predictions = predict(tree, testData);
disp('Test Verisi Tahminleri:');
disp(predictions);
real = {'No'; 'Yes'; 'No'};
acc = sum(strcmp(predictions, real)) / numel(real);
fprintf('accuracy: %.2f%%\n', acc * 100);





%%Entropi Function
function E = entropy_calc(labels)
    classes = unique(labels);
    N = length(labels);
    E = 0;
    for i = 1:length(classes)
        p = sum(strcmp(labels, classes{i})) / N;
        if p > 0
            E = E - p * log2(p);
        end
    end
end



%%Information Gain
function gain = information_gain(data, attribute, target)
    values = unique(data.(attribute));
    total_entropy = entropy_calc(data.(target));
    N = height(data);
    weighted_entropy = 0;
    for i = 1:length(values)  
        subset = data(strcmp(data.(attribute), values{i}), :);
        weighted_entropy = weighted_entropy + ...
                           (height(subset)/N) * entropy_calc(subset.(target));
    end
    gain = total_entropy - weighted_entropy;
end




%%ID3
function tree = id3(data, features, target)
    if length(unique(data.(target))) == 1
        tree.label = unique(data.(target));
        return
    end

    if isempty(features)
        tree.label = mode(categorical(data.(target)));
        return
    end

    gains = zeros(1, length(features));
    for i = 1:length(features)
        gains(i) = information_gain(data, features{i}, target);
    end
    [~, best_idx] = max(gains);
    best_feature = features{best_idx};

    tree.attribute = best_feature;
    values = unique(data.(best_feature));
    for i = 1:length(values)
        val = values{i};
        val_clean = matlab.lang.makeValidName(val);
        subset = data(strcmp(data.(best_feature), val), :);
        if isempty(subset)
            tree.branches.(val_clean).label = mode(categorical(data.(target)));
        else
            new_features = setdiff(features, best_feature);
            tree.branches.(val_clean) = id3(subset, new_features, target);
        end
    end
end



%%Predict ID -3

function label = predict_id3(tree, sample)
    if isfield(tree, 'label')
        label = tree.label;
    else
        attr = tree.attribute;
        val = sample.(attr);

        val_clean = matlab.lang.makeValidName(val);
        fieldName = char(val_clean);

        if isfield(tree.branches, fieldName)
            label = predict_id3(tree.branches.(fieldName), sample);
        else
            label = 'Unknown';
        end
    end
end



%%2.Örnek Kodu
features = {'Outlook', 'Temperature', 'Humidity', 'Wind'};
target = 'PlayTennis';
decision_tree = id3(data, features, target);

predictions = strings(height(data), 1);
for i = 1:height(data)
    sample = data(i, :);
    predictions(i) = string(predict_id3(decision_tree, sample));
end

true_labels = string(data.(target));

accuracy = sum(predictions == true_labels) / height(data);
fprintf('Eğitim verisi doğruluğu: %.2f%%\n', accuracy * 100);