function tcorr = compute_tcorr(nb_windows, nb_perm, nb_features, t_win, frame_indices, perms_sub1, sub1, sub2, perms_sub2)
tcorr = nan(nb_windows, nb_features, nb_perm, length(stats)*2);
for j = 1: nb_perm + 1
        hop_size = int32(t_win/3);
        frame_indices_slices_schema = get_frame_list_per_window(frame_indices, t_win, hop_size);
        
        sub1_perm = sub1(:, perms_sub1(j, :));
        sub2_perm = sub2(:, perms_sub2(j, :));

        for slice_i = 1 : length(frame_indices_slices_schema)

            sub1_slice = sub1_perm(:, frame_indices_slices_schema{slice_i});
            sub2_slice = sub2_perm(:, frame_indices_slices_schema{slice_i});

            tmp_corr = fast_corr( ...
                sub2_slice', ...
                sub2_slice');

            tcorr(slice_i, :, j, 1) = min()
            tcorr(slice_i, :, j, 2) = min()

        end
end
end