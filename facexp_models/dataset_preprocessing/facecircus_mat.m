clc; clear;
fps = 30;

num_neigh = 5; 
mean_knn = true;
method = 'pdist'; % supported 'l2' or 'pdist'
w_lens = int32((1:0.5:4) * fps);  % in seconds
%w_lens = 36090;

props = 0.4:0.1:0.8;
%props = 0;

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

for prop_i = 1:length(props)
fprintf("agreement proportion on peaks %d out of %d\n", prop_i, ...
    length(props))
prop_agreem = props(prop_i);
n_subjs = size(FaceRatings, 2);

% SELECT ONLY THE FRAMES WEHERE AT LEAST A CERTAIN NUMBER OF PARTICIPANTS
% AGREE ON A PEAK
t_indices = find(sum(squeeze(FaceRatingsProcessed(1,:,:)==100), 2) > n_subjs*prop_agreem);
frame_indices = cellfun(@(t) (t-2)*fps : (t+2)*fps, num2cell(t_indices), ...
    'UniformOutput', false);
frame_indices = unique(cell2mat(frame_indices));

data_framed = data(:, frame_indices, :, :);
%data_framed = data(:, :, :, :);

% COMPUTE 3-NN spatial non-overlapping mean for the data via pdist
if mean_knn
    features_3nn = pdist(squeeze(data_framed(1, 1, :, :)));
    features_3nn = squareform(features_3nn);
    
    j =1;
    for vx = 1 : size(data_framed,3)
        if all(isnan(features_3nn(vx, :) ))
            continue
        end
        [ ~,  vx3nn{j} ] = mink(features_3nn(vx, :), num_neigh+1);
        pos_vx3nn{j} = squeeze(data_framed(1, 1, vx3nn{j} , : ));
        features_3nn(vx3nn{j}, :) = nan;
        features_3nn(:, vx3nn{j}) = nan;
        j = j+1;
    end
    clear j;
    
    mean_pos_3nn = cellfun(@(pos) mean(pos,1), pos_vx3nn, 'UniformOutput', ...
        false);
    mean_pos_3nn = cat(1, mean_pos_3nn{:})';
    data_3nn = cellfun(@(indices) mean(data_framed(:, :, indices, :), 3), ...
        vx3nn, 'UniformOutput', false);
    data_3nn = cat(3, data_3nn{:});
    data_framed = data_3nn;
end

% preprocess data as L2 norm or pair-wise distance
if matches(method , 'l2')
    % A) INPUT DATA AS L2 NORM
    indata = sqrt(sum(data_framed(:, : , :, [1, 2]) .^ 2, 4));
elseif matches(method , 'pdist')
    % B) INPUT DATA AS PDIST
    indata = cell(size(data_framed, 1), size(data_framed, 2));
    cr = "";
    for k = 1: size(data_framed, 1)
        msg = compose("computing pdist for subj %d-th out of %d", ...
            k, size(data_framed, 1));
        fprintf(cr + msg)
        cr = repmat('\b', 1, strlength(msg));
        for t = 1: size(frame_indices, 1)
            indata{k, t} = pdist(squeeze(data_framed(k, t, :, :)));
        end
    end
    indata = cat(3, indata{:});
    indata = reshape(indata, size(indata, 1), ...
        size(indata, 2), size(data_framed, 1),  size(data_framed, 2) );
    indata = squeeze(permute(indata, [3, 4, 2, 1]));
    fprintf(cr)
end

% compute isc corr matrix for different windows length
% OUTPUT := [n_windlens, n_subj, n_subj, n_vertices]
[isc_corr, isc_pval] = compute_isc(indata, w_lens);


% corrections on p_values
alpha = 0.05;
mulcomp_method = true; fisher_method = false;

% (tippet method):
if mulcomp_method
    % correction for multiple comparisons
    num_feat_init = size(data_framed, 3);
    num_feat_pdist = (num_feat_init-1)*num_feat_init/2;
    num_comparisons = (num_feat_pdist + num_feat_init)* length(w_lens); %counting for both methods
    threshold = alpha / num_comparisons;

    % hard setting does not work 
    % every pair of (vertix,win) has at least one pari of subject
    % non-significant:
    survivors = isc_pval<threshold;
    % indicing the significant data
    isc_corr_nullnonsign = isc_corr;
    isc_corr_nullnonsign(~survivors) = nan;

end
if fisher_method
    reshaped_pvalues = reshape(isc_pval, size(isc_pval, 1), size(isc_pval, 2)*size(isc_pval, 3), size(isc_pval, 4) );
    chi_squaredist = -2*squeeze(sum(log(reshaped_pvalues),2));
    % fisher test:
    new_pval = 1 - chi2cdf(chi_squaredist, repmat(2*size(chi_squaredist, 2), size(chi_squaredist)));

    survivors = new_pval<alpha;
    survivors = reshape(survivors,size(survivors,1),1,1, size(survivors,2));
    survivors = repmat(survivors, 1, size(isc_corr,2), size(isc_corr,3), 1);
    % indicing the significant data
    isc_corr_nullnonsign = isc_corr;
    isc_corr_nullnonsign(~survivors) = nan;
end
% calculate the an the average isc
mean_subj_sign = squeeze(mean(isc_corr_nullnonsign, [2, 3]));

% get the vertices indices for the 20 highest correlation values
[values, indices] = maxk(mean_subj_sign, 60, 2);
%plot(mean_subj_sign(:,indices(1,1:10)))

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

template = h5read(datapath + 'h5out_30fps/local/l3celdsvq6yu4zufiag3rozjhmttv13n_output-001_local.h5', '/v');
template =  squeeze(mean(template, 3));
fig = figure('Visible', 0, 'Position',[0 0 2000 1000]);
for win_i = 1: length(w_lens)
subplot(ceil(sqrt(length(w_lens))), ceil(sqrt(length(w_lens))), win_i)
cmap = jet;
scatter(template(1, :), template(2, :));
hold on;
if matches(method, 'l2')
    for vx = 1:size(indices, 2)
        if isnan(values(win_i, vx))
            break
        end
        i = indices(win_i, vx);
        color_idx = round(interp1(linspace(-1, 1, size(cmap, 1)), 1:size(cmap, 1), values(win_i, vx)));
        if mean_knn
            scatter(mean_pos_3nn(1, i(1)), mean_pos_3nn(2, i(1)), 'MarkerFaceColor', cmap(color_idx, :))
        else
            scatter(template(1, i(1)), template(2, i(1)), 'MarkerFaceColor', cmap(color_idx, :))
        end
    end
elseif matches(method, "pdist")
    for vx = 1: size(indices, 2)
        if isnan(values(win_i, vx))
            break
        end
        ind2sqr = zeros(1, size(isc_corr, 4));
        ind2sqr(indices(win_i, vx))=1;
        [i, j] = find(squareform(ind2sqr));
        color_idx = round(interp1(linspace(-1, 1, size(cmap, 1)), 1:size(cmap, 1), values(win_i, vx)));
        if mean_knn
            line([mean_pos_3nn(1, i(1)), mean_pos_3nn(1, j(1))], [mean_pos_3nn(2, i(1)), mean_pos_3nn(2, j(1))], 'Color', cmap(color_idx, :), 'LineWidth', 1 );
        else
            line([template(1, i(1)), template(1, j(1))], [template(2, i(1)), template(2, j(1))], 'Color', cmap(color_idx, :), 'LineWidth', 1);
        end
    end
    
end
title(compose("win length: %.2f s", w_lens(win_i)/fps))
colormap(cmap);
colorbar;
hold off;
end
sgtitle(compose("agreement Proportion: %.1f", prop_agreem))
if mean_knn
    knn_fid = num_neigh;
else
    knn_fid = 0;
end
filename = compose("%s_win_%d_prop_%.1f_knn_%d.png", method, win_i, prop_agreem, knn_fid );
saveas(fig, fullfile(datapath,filename));
close(fig);

end

filename_suffix = 1;
filename= "isc_"+method+"_";
while exist(fullfile(datapath, filename+string(filename_suffix)))
    filename_suffix = filename_suffix+1;
end
save(fullfile(datapath, filename+string(filename_suffix)), "ISC", '-v7.3')


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

    isc_corr = nan(nb_windows, nb_subj, nb_subj, nb_features);
    isc_pval = nan(nb_windows, nb_subj, nb_subj, nb_features);

    cr = "";
    for win_j = 1:nb_windows
        msg = compose("t_window %d / %d", win_j, nb_windows);
        fprintf(cr + msg );
        cr = repmat('\b', 1, strlength(msg));

        t_win = w_lens(win_j);
        hop_size = int32(t_win );
        t_range = get_frame_list_per_window(timepoints, t_win, hop_size);

        % test: data_tmp = reshape(1:numel(data), size(data));  
        % win_avg_tmp = compute_win_avg(data_tmp, t_range);
        win_avg = compute_win_avg(data, t_range); % outputs subj by features by frame-set
        
        for n = 1 : size(win_avg, 2)
            [r, p] = corr(squeeze(win_avg(:, n, :))');
            isc_corr(win_j, :, :, n) = r;
            isc_pval(win_j, :, :, n) = p;
        end
    
    end
    fprintf(cr)
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
