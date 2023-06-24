%% classifier
clear; clc;
meddata = load('../mediapipe/ADFES_med/training_data.mat');
northdata = load('../mediapipe/ADFES_north/training_data.mat');
data.med = meddata.training_data; clear meddata;
data.north = northdata.training_data; clear northdata;

%%

med_emo = string({data.med.emo_id});
med_feat = cat(2, data.med.average_displacement);
north_emo = string({data.north.emo_id});
north_feat = cat(2, data.north.average_displacement);

labels = cat(2, north_emo, med_emo);
features = cat(2, north_feat, med_feat)';

rng(14051983)

n_folds = 5;
neighbors = 3:2:15;
n_neighbors = numel(neighbors);

optimal_neighbors = nan(n_folds,1);

kfold_partition = cvpartition(labels,'KFold',n_folds);

fold_loss = nan(n_neighbors,1);

confusion_matrices = cell(n_folds,1);

for k = 1:n_folds

    fprintf('Processing fold %d out of %d\n', k, n_folds)

    training_features = features(kfold_partition.training(k),:);
    test_features = features(kfold_partition.test(k),:);

    training_labels = labels(kfold_partition.training(k));
    test_labels = labels(kfold_partition.test(k));

    validation_partition = cvpartition(training_labels, "KFold", n_folds);

    for n = 1:n_neighbors

        knn_training=fitcknn(training_features,...
            training_labels,...
            'CVpartition',validation_partition,...
            'NumNeighbors',neighbors(n),...
            'DistanceWeight','inverse',...
            'BreakTies','nearest',...
            'Distance','euclidean');

        fold_loss(n) = kfoldLoss(knn_training);

    end

    [~, id_optimal_neighbors] = min(fold_loss);

    optimal_neighbors(k) = neighbors(id_optimal_neighbors);

    knn_model=fitcknn(training_features,...
        training_labels,...
        'NumNeighbors',neighbors(id_optimal_neighbors),...
        'DistanceWeight','inverse',...
        'BreakTies','nearest',...
        'Distance','euclidean');

    test_labels_predicted = predict(knn_model,test_features);

    confusion_matrices{k} = ...
        confusionmat(test_labels,test_labels_predicted);
    accuracy = sum(diag(confusion_matrices{k}))/size(test_features,1)

end