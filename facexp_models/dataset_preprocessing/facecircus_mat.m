dopca = 0;
pca_component = 20;
fps = 30;
w_lens = int32((1:0.5:4) * fps);  % in seconds
datapath = "/data1/EMOVIE_sampaolo/FACE/FaceCircus/data/";
datapath = "/home/matteo/Code/FACEXP/data/";
file2load = datapath + "data.mat";

if exist(file2load, 'file')
    load(file2load);
else
    [data, data_L2norm, data_L2_direction] = preprocess_h5_files(datapath);
    save(datapath + "data.mat", 'data');
    save(datapath + "data_L2norm.mat", 'data_L2norm');
end

data_L2norm = sqrt(sum(data(:, :, :, [1, 2]) .^ 2, 4));
data_pdist = pdist(reshape(data, [], size(data, 3)));
if isempty(w_lens)
    w_lens = int32((2:2:20) * fps);
end

indata = data_L2norm;
if dopca
    indata = zscore(indata, 0, 2);
    indata = get_pca(indata, pca_component);
end

[isc_corr, isc_pval] = compute_isc(indata, w_lens);
save(datapath + "isc_corr.mat", 'isc_corr');
save(datapath + "isc_pval.mat", 'isc_pval');


function [data, data_L2norm, direction_L2] = preprocess_h5_files(datapath)
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
    
    % Load vertices, filenames, and shapes from h5
    file_ids = {};
    meshes = {};
    shapes = {};
    in_folder = fullfile(datapath, 'h5out', 'local');
    
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

    for win_j = 1:nb_windows
        fprintf('\rt_window %d / %d', win_j, nb_windows);

        t_win = w_lens(win_j);
        hop_size = int32(t_win / 3);
        t_range = cell(floor((timepoints - t_win) / hop_size),1);
        
        for t = 1: floor((timepoints - t_win) / hop_size)
            init_t =  (t - 1) * hop_size + 1;
            t_range{t} = init_t : init_t + t_win;
        end
        
        for vx = 1:nb_features
            win_avg = squeeze(mean(data(:, t_range, vx), 2));
            [r,p] = corr(win_avg');
            isc_corr(win_j, :, :, vx) = r;
            isc_pval(win_j, :, :, vx) = p;
        end
    end

    for vx = 1:nb_features
        [r, p] = corr(squeeze(data(:, :, vx))');
        isc_corr(end, :, :, vx) = r;
        isc_pval(end, :, :, vx) = p;
    end

    fprintf('\n');
end

function data_out = get_pca(data, pca_component)
    data_reshaped = reshape(data, [], size(data, 3));
    pca_model = pca(data_reshaped, 'NumComponents', pca_component);
    
    data_out = nan(size(data, 1), size(data, 2), pca_component);
    for subj = 1:size(data, 1)
        data_out(subj, :, :) = pcatransform(pca_model, squeeze(data(subj, :, :)));
    end
end
