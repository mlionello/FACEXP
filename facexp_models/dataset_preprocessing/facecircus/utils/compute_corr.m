function isc_corr = compute_corr(nb_windows, nb_perm, nb_features, w_lens, frame_indices, perms_sub1, sub1, sub2, perms_sub2)
isc_corr = nan(nb_windows, 1, nb_features, nb_perm);
frame_indices_slices_schema = nan;
for win_j = 1: nb_windows
    t_win = w_lens(win_j);
    if t_win>0
        hop_size = int32(t_win/3);
        frame_indices_slices_schema = get_frame_list_per_window(frame_indices, t_win, hop_size);
        if length(frame_indices_slices_schema) < 2; continue; end
    end
    parfor j = 1: nb_perm + 1
        sub1_perm = sub1(:, perms_sub1(j, :));
        sub2_perm = sub2(:, perms_sub2(j, :));
        if t_win>0
            sub1_perm = compute_win_avg(sub1_perm, frame_indices_slices_schema);
            sub2_perm = compute_win_avg(sub2_perm, frame_indices_slices_schema);
        end

        isc_corr(win_j, 1, :, j) = fast_corr( ...
            sub1_perm', ...
            sub2_perm');
    end
end
end

function win_avg = compute_win_avg(data, frame_indices_schema)
    [n_rows, ~] = size(data);
    num_windows = numel(frame_indices_schema);

    win_avg = zeros(n_rows, num_windows);
    for i = 1: num_windows
        indices = frame_indices_schema{i};
        win_avg(:, i) = mean(data(:, indices), 2);
    end
end