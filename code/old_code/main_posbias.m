clear
clc

data_dir = '../mediapipe/ADFES_med';
filename_prefix = '.h5';

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

    temp_name = strsplit(file_names{f},'-');

    sub_id{f} = temp_name{1};

    emo_id{f} = temp_name{2};

    trial_id{f} = 1;

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

average_displacement = ...
    cellfun(@(x) mean(x,2),distances,'UniformOutput',false);

training_data = struct('sub_id', sub_id, 'emo_id', emo_id, 'trial_id', trial_id, ...
    'average_displacement', average_displacement);
save(fullfile(data_dir, 'training_data'), 'training_data');