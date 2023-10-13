clc; clear; addpath('utils');
fps = 30;

num_neigh = 5; 
method = 'pdist'; % supported 'l2' or 'pdist'
w_lens = int32((1:0.5:6) * fps);  % in seconds
w_lens = [0, w_lens];
w_lens = int32((4:6:40) * fps);  % in seconds

props = 0.4:0.1:0.8;
props = 0;

datapath = "/data1/EMOVIE_sampaolo/FACE/FaceCircus/data/";
% datapath = "/home/matteo/Code/FACEXP/data/";
file2load = datapath + "data.mat";

% Import metadata
FaceRatings = load(fullfile(datapath, 'FaceRatings.mat'));
FaceRatings = FaceRatings.FaceRatings;

FaceRatingsMetaData = load(fullfile(datapath, 'FaceRatingsMetaData.mat'));
FaceRatingsMetaData = FaceRatingsMetaData.FaceRatingsMetaData;

FaceRatingsOverlap = load(fullfile(datapath, 'FaceRatingsOverlap.mat'));
FaceRatingsOverlap = FaceRatingsOverlap.FaceRatingsOverlap;

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
    data_file_path = fullfile(outpath, h5_files{j}(1: end-3)); 
    if exist(data_file_path +'.mat', 'file')
        fprintf("loading data...")
        file_ids{j} = data_file_path + '.mat';
        if matches(method, 'pdist')
            load(data_file_path+'_meanpos')
        end
        fprintf(" done\n")
    else
        [data_in, file_ids{j}] = preprocess_h5_files(in_folder, outpath, h5_files{j}, num_neigh, method, data_file_path);
    end
end
load(fullfile(outpath, 'metadata'));

for prop_i = 1:length(props)
fprintf("agreement proportion on peaks %d out of %d\n", prop_i, ...
    length(props))
prop_agreem = props(prop_i);
n_subjs = size(FaceRatings, 2);

if prop_agreem>0
    % SELECT ONLY THE FRAMES WEHERE AT LEAST A CERTAIN NUMBER OF PARTICIPANTS
    % AGREE ON A PEAK
    t_indices = find(sum(squeeze(FaceRatingsProcessed(1, :, :)==100), 2) > n_subjs*prop_agreem);
    frame_indices = cellfun(@(t) (t-2)*fps : (t+2)*fps, num2cell(t_indices), ...
        'UniformOutput', false);
    frame_indices = unique(cell2mat(frame_indices));
else 
    frame_indices = 1: size(FaceRatingsProcessed, 2)*fps;
end

% compute isc corr matrix for different windows length
% OUTPUT := [n_windlens, n_subj, n_subj, n_vertices]
if false % exist(fullfile(outpath, 'isc'), 'file')
    load(fullfile(outpath, 'isc'))
else
    isc_corr = compute_isc(file_ids, w_lens, frame_indices, metadata);
end

% calculate the an the average isc
isc_corr_mean = mean(isc_corr, [2, 3]);
isc_corr_mean = atanh(isc_corr_mean);
isc_corr_mean = permute(isc_corr_mean, [5, 4, 1, 2, 3]); % perms by feat by wins

fw_max = max(isc_corr_mean(2: end, :, :), [], 2);

clear pval_corrected
for j = 1: size(fw_max, 3)
    concat_unc = cat(1, isc_corr_mean(1, :, j), ...
        repmat(fw_max(:, :, j), 1, size(isc_corr_mean, 2), 1));
    pval_corrected{j} = tiedrank(-concat_unc, 0, 1)/size(concat_unc, 1);
end
pval_corrected = cat(3, pval_corrected{:});

alpha = 0.05;
pval_corrected = pval_corrected < alpha/2;

mean_subj_sign = isc_corr_mean;
mean_subj_sign(~pval_corrected)=nan;

% get the vertices indices for the 20 highest correlation values
[values, indices] = maxk(mean_subj_sign(1,:,:), 20, 2);
%plot(mean_subj_sign(:, indices(1, 1:10)))

ISC.method = method;
ISC.prop_agreem{prop_i} = prop_agreem;
ISC.w_lens = w_lens;
ISC.isc_corr{prop_i} = isc_corr;
ISC.pval_corrected{prop_i} = pval_corrected;
ISC.indices_highest_corr{prop_i} = squeeze(indices);
ISC.values_highest_corr{prop_i} = squeeze(values);
ISC.subj_id = subj_id;
ISC.alpha = alpha;
ISC.fw_max = fw_max;
ISC.prop_agreem = prop_agreem;
ISC.isc_corr_mean = isc_corr_mean;
ISC.mean_subj_sign = mean_subj_sign;
ISC.n_subjs = n_subjs;
ISC.mean_pos_3nn = mean_pos_3nn;

plot_res_and_save(mean_pos_3nn, values, indices, w_lens, method, fps, prop_agreem, outpath, num_neigh)

fig=figure('Visible',0); subplot(212); plot(squeeze(nanmean(mean_subj_sign(1, :, :), 2)))
subplot(211); plot(squeeze(mean_subj_sign(1, :, :))'); 
filename = compose("win_increm_signi_corr.png" );
saveas(fig, fullfile(outpath,filename));
close(fig);

filename_suffix = 1;
filename = "isc_struct";
while exist(fullfile(outpath, filename+string(filename_suffix)))
    filename_suffix = filename_suffix+1;
end
save(fullfile(outpath, filename+string(filename_suffix)), "ISC", '-v7.3')
end

