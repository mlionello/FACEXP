function facecircus(datapath)
addpath('utils');
fps = 30;
corr_method = 'tcorr';
w_lens = 5*fps;

num_neigh = 5;
method = 'l2'; % supported 'l2' or 'pdist'

prop_agreem = 0;

% datapath = ".../FACE/FaceCircus/data/";

% Import metadata
FaceRatings = load(fullfile(datapath, 'FaceRatings.mat'));
FaceRatings = FaceRatings.FaceRatings;

FaceRatingsMetaData = load(fullfile(datapath, 'FaceRatingsMetaData.mat'));
FaceRatingsMetaData = FaceRatingsMetaData.FaceRatingsMetaData;

% FaceRatingsOverlap = load(fullfile(datapath, 'FaceRatingsOverlap.mat'));
% FaceRatingsOverlap = FaceRatingsOverlap.FaceRatingsOverlap;

FaceRatingsProcessed = load(fullfile(datapath, 'FaceRatingsProcessed.mat'));
FaceRatingsProcessed = FaceRatingsProcessed.FaceRatingsProcessed;

% Extract subj_id code
subj_id = string(FaceRatingsMetaData.SubjectsID);

file_ids = {};
outpath = fullfile(datapath, method, sprintf("knn_%d", num_neigh));
if ~exist(outpath, 'dir')
    mkdir(outpath)
end
in_folder = fullfile(datapath, 'h5out_30fps', 'local'); 
h5_files = {dir(fullfile(in_folder, '*.h5')).name}; 
for j = 1:length(h5_files)
    data_file_path = fullfile(outpath, 'subjects', h5_files{j}(1: end-3)); 
    if exist(data_file_path +'.mat', 'file')
        fprintf("loading data...")
        file_ids{j} = data_file_path + '.mat';
        fprintf(" done\n")
    else
        if ~exist(fullfile(outpath, 'subjects'), 'dir')
            mkdir(fullfile(outpath, 'subjects'))
        end
        [~, file_ids{j}] = preprocess_h5_files(in_folder, fullfile(outpath, 'subjects'), h5_files{j}, num_neigh, method, data_file_path);
    end
end
if num_neigh>0
    load(data_file_path+'_meanpos', 'mean_pos_3nn')
end
load(fullfile(outpath, 'subjects', 'metadata'), 'metadata');

n_subjs = size(FaceRatings, 3);

if prop_agreem>0
    % SELECT ONLY THOSE FRAMES WEHERE AT LEAST A CERTAIN NUMBER OF
    % PARTICIPANTS AGREE ON A PEAK
    t_indices = find(sum(squeeze(FaceRatingsProcessed(1, :, :)==100), 2) > n_subjs*prop_agreem);
    frame_indices = cellfun(@(t) (t-2)*fps : (t+2)*fps, num2cell(t_indices), ...
        'UniformOutput', false);
    frame_indices = unique(cell2mat(frame_indices))';
else 
    frame_indices = 60: size(FaceRatingsProcessed, 2)*fps;
end

% compute isc corr matrix for different windows length
% OUTPUT := [n_windlens, n_subj, n_subj, n_vertices]
isc_corr = isc_pairwise_sampling(file_ids, w_lens, frame_indices, metadata, outpath, corr_method);

% calculate the an the average isc
if matches(corr_method, 'corr')
    isc_corr_mean = mean(isc_corr, 2);
    isc_corr_mean = permute(isc_corr_mean, [ 4, 3, 1, 2]); % perms by feat by wins
elseif matches(corr_method, 'tcorr')
    isc_corr_mean = isc_corr;
    isc_corr_mean = permute(isc_corr_mean, [ 3, 1, 2]); % perms by feat by wins
end
nb_fetures = size(isc_corr_mean, 2);
nb_windows = size(isc_corr_mean, 3);
nb_perm = size(isc_corr_mean, 1)-1;

%ISC.isc_corr = isc_corr;
ISC.method = method;
ISC.corr_method = corr_method;
ISC.w_lens = w_lens;
ISC.isc_corr_mean = isc_corr_mean;
ISC.subj_id = subj_id;
ISC.prop_agreem = prop_agreem;
ISC.n_subjs = n_subjs;
ISC.mean_pos_3nn = mean_pos_3nn;
ISC.outpath = outpath;
ISC.datapath = datapath;
ISC.fps = fps;
ISC.num_neigh = num_neigh;
ISC.nb_fetures = nb_fetures;
ISC.nb_windows = nb_windows;
ISC.nb_perm = nb_perm;

filename_suffix = 1;
filename = compose("isc_struct_wlen_%d_pa_%d_method_%s", ...
    w_lens, prop_agreem, corr_method);
while exist(fullfile(outpath, filename+string(filename_suffix)+'.mat'), 'file')
    filename_suffix = filename_suffix + 1;
end
save(fullfile(outpath, filename+string(filename_suffix)), "ISC", '-v7.3')
end