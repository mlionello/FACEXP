clc; clear;
method = 'pdist';
dopca = 0;
pca_component = 20;
fps = 30;
w_lens = int32((1:0.5:4) * fps);  % in seconds
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

if exist(file2load, 'file')
    fprintf("loading data...")
    load(file2load);
    fprintf(" done\n")
else
    [data, data_L2norm, data_L2_direction, file_ids] = preprocess_h5_files(datapath);
    save(datapath + "data.mat", 'data');
    save(datapath + "data_L2norm.mat", 'data_L2norm');
    save(datapath + "file_ids.mat", 'file_ids');
end

props = 0.4:0.1:0.8;
for prop_i = 1:length(props)
fprintf("agreement proportion on peaks %d out of %d\n", prop_i, length(props))
prop_agreem = props(prop_i);
n_subjs = size(FaceRatings, 2);

% SELECT ONLY THE FRAMES WEHERE AT LEAST A CERTAIN NUMBER OF PARTICIPANTS
% AGREE ON A PEAK
t_indices = find(sum(squeeze(FaceRatingsProcessed(1,:,:)==100),2)>n_subjs*prop_agreem);
frame_indices = cellfun(@(t) (t-2)*fps : (t+2)*fps, num2cell(t_indices), 'UniformOutput', false);
frame_indices = unique(cell2mat(frame_indices));

if matches(method , 'l2')
    % A) INPUT DATA AS L2 NORM
    indata = sqrt(sum(data(:, frame_indices , :, [1, 2]) .^ 2, 4));
elseif matches(method , 'pdist')
    % B) INPUT DATA AS PDIST
    indata = cell(size(data, 1), size(frame_indices, 1));
    cr = "";
    for k =1:size(data, 1)
        msg = compose("computing pdist for subj %d-th out of %d", k, size(data, 1));
        fprintf(cr + msg)
        cr = repmat('\b', 1, strlength(msg));
        for t = 1:size(frame_indices, 1)
            indata{k,t} = pdist(squeeze(data(k, frame_indices(t), :, :)));
        end
    end
    indata = cat(3, indata{:});
    indata = reshape(indata, 1, size(data, 3)*(size(data, 3)-1)/2, size(data,1), length(frame_indices));
    indata = squeeze(permute(indata, [3, 4, 2, 1]));
    fprintf(cr)
end

if dopca
    indata = zscore(indata, 0, 2);
    indata = get_pca(indata, pca_component);
end

% compute isc corr matrix for different windows length
% OUTPUT := [n_windlens, n_subj, n_subj, n_vertices]
[isc_corr, isc_pval] = compute_isc(indata, w_lens);

% correction for multiple comparisons
survivors = isc_pval<0.05/size(indata,2)/length(w_lens);

% indicing the significant data
isc_corr_nullnonsign = isc_corr;
isc_corr_nullnonsign(~survivors)=0;

% calculate the an the average isc
mean_subj_sign = mean(isc_corr_nullnonsign, [2, 3]);

% get the vertices indices for the 10 highest correlation values
[values, indices] = maxk(mean_subj_sign, 10, 4);

ISC.method = method;
ISC.prop_agreem{prop_i} = prop_agreem;
ISC.w_lens = w_lens;
ISC.pca_comp = pca_component;
ISC.dopca = dopca;
ISC.isc_corr{prop_i} = isc_corr;
ISC.isc_pval{prop_i} = isc_pval;
ISC.indices_highest_corr{prop_i} = squeeze(indices);
ISC.values_highest_corr{prop_i} = squeeze(values);
ISC.subj_id = subj_id;
end

filename_suffix = 1;
filename= "isc_"+method+"_";
while exists(fullfile(datapath, filename+string(filename_suffix)))
    filename_suffix = filename_suffix+1;
end
save(fullfile(datapath, filename+string(filename_suffix)), "ISC")

function [data, data_L2norm, direction_L2, file_ids] = preprocess_h5_files(datapath)
    
    % Load vertices, filenames, and shapes from h5
    file_ids = {};
    meshes = {};
    shapes = {};
    in_folder = fullfile(datapath, 'h5out_30fps', 'local');
    
    h5_files = dir(fullfile(in_folder, '*.h5'));
    
    for j = 1:length(h5_files)
        fileh5 = h5_files(j);
        fprintf('loading: %s\n', fileh5.name);
        
        data_in = h5read(fullfile(in_folder, fileh5.name), '/v');
        
        file_ids{end+1} = fullfile(in_folder, fileh5.name);
        meshes{end+1} = data_in;
        shapes{end+1} = size(data_in,3);
    end

    % Get minimum t len among subjects (should vary just in the order of tens of ms)
    min_t = min(cell2mat(shapes));

    % Prepare subj baseline vertices array
    data = cellfun(@(mesh) mesh( :, :, 1:min_t), meshes, 'UniformOutput', false);
    data = cat(4, data{:});
    data = permute(data, [4, 3, 2, 1]);

    % Compute data_L2norm and direction_L2
    data_L2norm = squeeze(sqrt(sum(data(:, :, :, [1, 2]) .^ 2, 4)));  % only x and y axis
    direction_L2 = data( :, :, :, [1, 2]) ./ data_L2norm;
end

function [isc_corr, isc_pval] = compute_isc(data, w_lens)
    [nb_subj, timepoints, nb_features] = size(data);
    nb_windows = length(w_lens);

    isc_corr = nan(nb_windows + 1, nb_subj, nb_subj, nb_features);
    isc_pval = nan(nb_windows + 1, nb_subj, nb_subj, nb_features);

    cr = "";
    for win_j = 1:nb_windows
        msg = compose('t_window %d / %d', win_j, nb_windows);
        fprintf(cr + msg );
        cr = repmat('\b', 1, strlength(msg));

        t_win = w_lens(win_j);
        hop_size = int32(t_win );
        t_range = get_frame_list_per_window(timepoints, t_win, hop_size);

        win_avg = compute_win_avg(data, t_range);
        
        for n = 1 : size(win_avg,3)
            [r, p] = corr(squeeze(win_avg(:,:,n))');
            isc_corr(win_j, :, :, vx) = r;
            isc_pval(win_j, :, :, vx) = p;
        end
        frpintf(cr)
    
    end
end

function t_range = get_frame_list_per_window(timepoints, t_win, hop_size)
        t_range = cell(floor((timepoints - t_win) / hop_size),1);
        
        for t = 1: floor((timepoints - t_win) / hop_size)
            init_t =  (t - 1) * hop_size + 1;
            t_range{t} = init_t : init_t + t_win;
        end
end

function win_avg = compute_win_avg(data, t_range)
        win_avg = cellfun(@(indices) mean(data(:, indices, :), 2), t_range, 'UniformOutput', false);
        win_avg = cat(4, win_avg{:});
        win_avg = squeeze(win_avg);
end

function data_out = get_pca(data, pca_component)
    data_reshaped = reshape(data, [], size(data, 3));
    pca_model = pca(data_reshaped, 'NumComponents', pca_component);
    
    data_out = nan(size(data, 1), size(data, 2), pca_component);
    for subj = 1:size(data, 1)
        data_out(subj, :, :) = pcatransform(pca_model, squeeze(data(subj, :, :)));
    end
end
