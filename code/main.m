clear
clc

data_dir = '../mediapipe/pilot';
filename_prefix = '_mediapipe.h5';

file_list = dir(strcat(data_dir,'/*',filename_prefix));

file_names = {file_list.name};

folder_names = {file_list.folder};

n_files = numel(file_names);

data = cell(n_files,1);

distances = cell(n_files,1);

n_vertices = cell(n_files,1);

n_dimensions = cell(n_files,1);

n_timepoints = cell(n_files,1);

sub_id = cell(n_files,1);

emo_id = cell(n_files,1);

trial_id = cell(n_files,1);

for f = 1:n_files

    fprintf('processing %d out of %d\n',f,n_files)

    file_to_import = strcat(folder_names{f},'/',file_names{f});

    temp_name = strsplit(file_names{f},'_');

    sub_id{f} = temp_name{1};

    emo_id{f} = temp_name{2}(1:2);

    trial_id{f} = temp_name{2}(3:4);

    temp_data = ...
        h5read(file_to_import,'/v');

    temp_data = permute(temp_data,[2,1,3]);

    n_vertices{f} = size(temp_data,1);

    n_dimensions{f} = size(temp_data,2);

    n_timepoints{f} = size(temp_data,3);

    distance_matrix = nan(n_vertices{f}*(n_vertices{f}-1)/2,n_timepoints{f});

    distance_ref = pdist(temp_data(:,:,1));

    for t = 1:n_timepoints{f}

        distance_matrix(:,t) = pdist(temp_data(:,:,t)) - distance_ref;

    end

    data{f} = temp_data;

    distances{f} = distance_matrix;

end

sub_id = categorical(sub_id);

emo_id = categorical(emo_id);

trial_id = categorical(trial_id);


average_displacement = ...
    cellfun(@(x) mean(x,2),distances,'UniformOutput',false);


%% classifier

rng(14051983)

features = cat(2,average_displacement{:})';
labels = emo_id;
n_folds = 5;
neighbors = 3:2:15;
n_neighbors = numel(neighbors);

optimal_neighbors = nan(n_folds,1);

kfold_partition = cvpartition(emo_id,'KFold',n_folds);

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

end

%figure;imagesc(squareform(mean(distances{11},2))); caxis([-0.5 0.5])