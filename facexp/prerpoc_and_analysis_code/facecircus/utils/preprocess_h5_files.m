function [indata, file_id] = preprocess_h5_files(in_folder, out_folder, fileh5, num_neigh, method, data_file_path)
    fprintf('loading: %s\n', fileh5);
    if exist(fullfile(out_folder, 'metadata.mat'))
        load(fullfile(out_folder, 'metadata'))
    else
        metadata = struct();
        metadata.timepoints = [];
        metadata.nb_features = [];
    end
    data = h5read(fullfile(in_folder, fileh5), '/v');
    file_id = fullfile(in_folder, fileh5);

    % COMPUTE 3-NN spatial non-overlapping mean for the data via pdist
    if num_neigh>0
        [data, mean_pos_3nn] = get_knnmean(data, num_neigh);
        save(data_file_path+'_meanpos', 'mean_pos_3nn')
    end

    % preprocess data as L2 norm or pair-wise distance
    if matches(method , 'l2')
        % A) INPUT DATA AS L2 NORM
        indata = squeeze(sqrt(sum(data( [1, 2] , :, :) .^ 2, 1)));
    elseif matches(method , 'pdist')
        % B) INPUT DATA AS PDIST
        indata = cell(size(data, 3), 1);
        cr = "";
        for t = 1: size(data, 3)
            msg = compose("computing pdist for t %d-th out of %d", ...
                t, size(data, 3));
            fprintf(cr + msg)
            cr = repmat('\b', 1, strlength(msg));
            indata{ t} = pdist(transpose(squeeze(data( :, :, t))));
        end
        indata = cat(1, indata{:});
        indata = transpose(indata); % features by timepoints
        fprintf(cr)
    end
    save(data_file_path, 'indata', '-v7.3')
    metadata.timepoints = [metadata.timepoints, size(indata, 2)];
    metadata.nb_features = [metadata.nb_features, size(indata, 1)];
    save(fullfile(out_folder, 'metadata'), 'metadata');
end

function [data_3nn, mean_pos_3nn]  = get_knnmean(data_framed, num_neigh)
    features_3nn = pdist(transpose(squeeze(data_framed(:, :, 1))));
    features_3nn = squareform(features_3nn);
    
    j = 1;
    for vx = 1 : size(features_3nn, 1)
        if all(isnan(features_3nn(vx, :)))
            continue
        end
        [ ~,  vx3nn{j} ] = mink(features_3nn(vx, :), num_neigh+1);
        pos_vx3nn{j} = squeeze(data_framed(:, vx3nn{j} , 1));
        features_3nn(vx3nn{j}, :) = nan;
        features_3nn(:, vx3nn{j}) = nan;
        j = j+ 1;
    end
    clear j;
    
    mean_pos_3nn = cellfun(@(pos) mean(pos,2), pos_vx3nn, 'UniformOutput', ...
        false);
    mean_pos_3nn = cat(2, mean_pos_3nn{:})';
    data_3nn = cellfun(@(indices) mean(data_framed(:, indices, :), 2), ...
        vx3nn, 'UniformOutput', false);
    data_3nn = cat(2, data_3nn{:});
end
