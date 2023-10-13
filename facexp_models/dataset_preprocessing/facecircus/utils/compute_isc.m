function [isc_corr] = compute_isc(file_ids, w_lens, t_frames, metadata)
    nb_perm = 200;
    nb_subj = length(file_ids);
    timepoints = min(min(metadata.timepoints), length(t_frames)); % this must be time_frames
    t_frames = t_frames(t_frames<=timepoints);
    nb_features = metadata.nb_features(1);

    nb_windows = length(w_lens);

    isc_corr = nan(nb_windows, nb_subj, nb_subj, nb_features);
    isc_corr_perm = nan(nb_windows, nb_subj, nb_subj, nb_features, nb_perm);
    
    frames_perm_schema{1} = 1:timepoints;
    for j = 2:nb_perm+1
        frames_perm_schema{j} = randperm(timepoints);
    end

    cr = "";
    for sub_i = 1: nb_subj-1
        sub1 = load(file_ids{sub_i});
        sub1 = sub1.indata(:, t_frames);
        for sub_j = sub_i+1: nb_subj
            sub2 = load(file_ids{sub_j});
            sub2 = sub2.indata(:, t_frames);
            for win_j = 1: nb_windows
%                 msg = compose("correlating subj %d with subj %d for t_window %d / %d", sub_i, sub_j, win_j, nb_windows);
%                 fprintf(cr + msg );
%                 cr = repmat('\b', 1, strlength(msg));
        
                t_win = w_lens(win_j);
                if t_win>0
                    hop_size = int32(t_win/3);
                    t_range = get_frame_list_per_window(timepoints, t_win, hop_size);
                    if length(t_range)<2
                        continue
                    end
                
                    % test: data_tmp = reshape(1:numel(data), size(data));
                    % win_avg_tmp = compute_win_avg(data_tmp, t_range);
                    % win_avg = compute_win_avg(data, t_range); % outputs subj by features by frame-set
                    sub_proc1 = compute_win_avg(sub1, t_range);
                    sub_proc2 = compute_win_avg(sub2, t_range);
                else
                    sub_proc1 = sub1;
                    sub_proc2 = sub2;
                end
                perms_ind = cellfun(@(item) item(item <= size(sub_proc2, 2)), frames_perm_schema, 'UniformOutput', false);
                perms_ind = cat(1, perms_ind{:});

                for j = 1: nb_perm+1
                    msg = compose("correlating subj %d with subj %d for t_window %d / %d; perm %d", sub_i, sub_j, win_j, nb_windows, j);
                    fprintf(cr + msg );
                    cr = repmat('\b', 1, strlength(msg));
                    for n = 1 : size(sub_proc1, 1)
                        isc_corr(win_j, sub_i, sub_j, n, j) = corr( ...
                            sub_proc1(n, perms_ind(1, :))', ...
                            sub_proc2(n, perms_ind(j, :))');
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

function t_range = get_frame_list_per_window(timepoints, t_win, hop_size)
        t_range = cell(floor((timepoints - t_win) / hop_size),1);
        
        for t = 1: floor((timepoints - t_win) / hop_size)
            init_t =  (t - 1) * hop_size + 1;
            t_range{t} = init_t : init_t + t_win;
        end
end

function win_avg = compute_win_avg(data, t_range)
        if length(size(data))==2
            data = reshape(data, 1, size(data,1), size(data,2));
        end
        win_avg = cellfun(@(indices) mean(data(:, :, indices), 3), t_range, 'UniformOutput', false);
        win_avg = cat(4, win_avg{:});
        win_avg = squeeze(win_avg);
end