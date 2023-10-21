function [isc_corr] = isc_pairwise_sampling(file_ids, w_lens, frame_indices, metadata, outpath, method)
    nb_perm = 200;

    file_ids(2) = [];
    metadata.timepoints(2) = [];
    metadata.nb_features(2) = [];
    
    nb_subj = length(file_ids);
    nb_comb_subj = int32(nb_subj*(nb_subj-1)/2);

    frame_indices = frame_indices(frame_indices <= min(metadata.timepoints));
    nb_features = metadata.nb_features(1);
    ws.nb_subj = nb_subj; ws.nb_features = nb_features; ws.files_ids = file_ids;
    ws.metadata = metadata; ws.nb_comb_subj = nb_comb_subj; ws.nb_perm = nb_perm;
    
    for n = 1: nb_subj
        for j = 1: nb_perm + 1
            if j == 1
                frames_perm_schema{n, j} = frame_indices;
                continue
            end
            frames_perm_schema{n, j} = frame_indices(randperm(length(frame_indices)));
        end
    end
    ws.frames_perm_schema = frames_perm_schema;

    if matches(method, 'corr')
        nb_windows = length(w_lens);
        isc_corr = nan(nb_windows, nb_comb_subj, nb_features, nb_perm+1);
    elseif matches(method, 'tcorr')
        nb_windows = length(get_frame_list_per_window(frame_indices, w_lens, int32(w_lens/3)));
        stats = ['min', 'max', 'mean', 'std', '95perc', 'pval_unc'];
        isc_corr = zeros(nb_windows, nb_features);
        ws.stats = stats;
    end
    ws.nb_windows = nb_windows;

    if ~exist(fullfile(outpath, 'checkouts'), 'dir')
        mkdir(fullfile(outpath, 'checkouts'));
    end
    save(fullfile(outpath, 'checkouts', 'ws_checkout'), 'ws', '-v7.3')
    clear ws

    cr = "";
    pair_counter = 1;
    tic;
    for sub_i = 1: nb_subj-1
        sub1 = load(file_ids{sub_i});
        sub1 = sub1.indata;
        perms_sub1 = cat(1, frames_perm_schema{sub_i, :}); % nb_perm by nb_t
        for sub_j = sub_i + 1: nb_subj
            msg = compose("correlating subj %d with subj %d; " + ...
                "pair counter %d out of %d; elapsed time: %d seconds", ...
                sub_i, sub_j, pair_counter, nb_comb_subj, int32(toc));
            fprintf(cr + msg );
            cr = repmat('\b', 1, strlength(msg));

            sub2 = load(file_ids{sub_j});
            sub2 = sub2.indata; % nb_vx by nb_t
            perms_sub2 = cat(1, frames_perm_schema{sub_j, :}); % nb_perm by nb_t

            if matches(method, 'corr')
                isc_corr(:, pair_counter, :, :) = compute_corr( ...
                    nb_windows, nb_perm, nb_features, ...
                    w_lens, frame_indices, perms_sub1, sub1, sub2, ...
                    perms_sub2);
            elseif matches(method, 'tcorr')
                tcorr = compute_tcorr( ...
                    nb_windows, nb_perm, nb_features, ...
                    w_lens, frame_indices, perms_sub1, sub1, sub2, ...
                    perms_sub2);
                tcorr = cat(2, tcorr{:});
                tcorr = reshape(tcorr, ...
                    nb_features, nb_windows, nb_perm+1);
                if pair_counter == 1
                    isc_corr = tcorr/single(nb_comb_subj);
                else
                    isc_corr = isc_corr + tcorr/single(nb_comb_subj);
                end
            end
            if mod(pair_counter, 50) == 0
                checkout_id = compose('isc_corr_checkout_sub_%02d_subcomb_%03d', ...
                    sub_i, pair_counter);
                save(fullfile(outpath, 'checkouts', checkout_id), ...
                    'isc_corr', '-v7.3')
            end
            pair_counter = pair_counter + 1;

        end
    
    end
    fprintf(cr)
end
