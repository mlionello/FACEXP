function [isc_corr] = compute_isc(file_ids, w_lens, frame_indices, metadata)
    nb_perm = 200;
    nb_subj = length(file_ids);

    file_ids(2) = [];
    metadata.timepoints(2) = [];
    metadata.nb_features(2) = [];

    frame_indices = frame_indices(frame_indices <= min(metadata.timepoints));
    nb_features = metadata.nb_features(1);

    nb_windows = length(w_lens);

    isc_corr = nan(nb_windows, nb_subj, nb_subj, nb_features);
    isc_corr_perm = nan(nb_windows, nb_subj, nb_subj, nb_features, nb_perm);
    
    for n = 1: nb_subj
        for j = 1: nb_perm+1
            if j == 1
                frames_perm_schema{n, j} = frame_indices;
                continue
            end
            frames_perm_schema{n, j} = frame_indices(randperm(length(frame_indices)));
        end
    end

    cr = "";
    for sub_i = 1: nb_subj-1
        sub1 = load(file_ids{sub_i});
        sub1 = sub1.indata;
        perms_sub1 = cat(1, frames_perm_schema{sub_i, :}); % nb_perm by nb_t
        for sub_j = sub_i + 1: nb_subj
            sub2 = load(file_ids{sub_j});
            sub2 = sub2.indata; % nb_vx by nb_t
            perms_sub2 = cat(1, frames_perm_schema{sub_j, :}); % nb_perm by nb_t
            for win_j = 1: nb_windows
                t_win = w_lens(win_j);
                if t_win>0
                    hop_size = int32(t_win/3);
                    frame_indices_slices_schema = get_frame_list_per_window(frame_indices, t_win, hop_size);
                            if length(frame_indices_slices_schema)<2; continue; end
                end
                for j = 1: nb_perm + 1
                    msg = compose("correlating subj %d with subj %d for " + ...
                        "t_window %d / %d; perm %d", sub_i, sub_j, win_j, ...
                        nb_windows, j);
                    fprintf(cr + msg );
                    cr = repmat('\b', 1, strlength(msg));
                    sub1_perm = sub1(:, perms_sub1(j, :));
                    sub2_perm = sub2(:, perms_sub2(j, :));
                    if t_win>0

                        % test: data_tmp = reshape(1:numel(data), size(data));
                        % win_avg_tmp = compute_win_avg(data_tmp, t_range);
                        % win_avg = compute_win_avg(data, t_range); % outputs subj by features by frame-set
                        sub1_perm = compute_win_avg(sub1_perm, frame_indices_slices_schema);
                        sub2_perm = compute_win_avg(sub2_perm, frame_indices_slices_schema);
                    end

                    for n = 1 : size(sub1_perm, 1)
                        isc_corr(win_j, sub_i, sub_j, n, j) = corr( ...
                            sub1_perm(n, :)', ...
                            sub2_perm(n, :)');
                        isc_corr(win_j, sub_j, sub_i, n, j) = isc_corr(win_j, sub_i, sub_j, n, j);
                        isc_corr(win_j, sub_j, sub_j, n, j) = 1;
                        isc_corr(win_j, sub_i, sub_i, n, j) = 1;
                    end
            
                end
            end
        end
    end
    fprintf(cr)
end

function frame_indices_slices = get_frame_list_per_window(frame_indices, t_win, hop_size)
        timepoints = length(frame_indices);
        frame_indices_slices = cell(floor((timepoints - t_win) / hop_size),1);
        
        for t = 1: floor((timepoints - t_win) / hop_size)
            init_t =  (t - 1) * hop_size + 1;
            frame_indices_slices{t} = init_t : init_t + t_win;
        end
end

function win_avg = compute_win_avg(data, frame_indices_schema)
        win_avg = cellfun(@(indices) mean(data( :, indices), 2), frame_indices_schema, 'UniformOutput', false);
        win_avg = cat(2, win_avg{:});
        win_avg = squeeze(win_avg);
end