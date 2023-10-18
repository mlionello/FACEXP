function tcorr = compute_tcorr(nb_windows, nb_perm, nb_features, t_win, frame_indices, perms_sub1, sub1, sub2, perms_sub2)
hop_size = int32(t_win/3);
tcorr = nan(nb_windows, nb_features, nb_perm, length(stats)*2);
frame_indices_slices_schema = get_frame_list_per_window(frame_indices, t_win, hop_size);
non_perm1 = sub1(:, perms_sub1(1, :));
non_perm2 = sub2(:, perms_sub2(1, :));

for j = 1: nb_perm + 1
        perm1 = sub1(:, perms_sub1(j, :));
        perm2 = sub2(:, perms_sub2(j, :));

        [~, ind] = sort( perm1( wins(x, :)));
        perms_sub1(1, wins(x, ind))

        for slice_i = 1 : length(frame_indices_slices_schema)

            t_perm = frame_indices_slices_schema{slice_i};
            [~, ind] = sort( perm1( t_perm));
            perm1_slice = non_perm1(t_perm(ind));

            t_perm = frame_indices_slices_schema{slice_i};
            [~, ind] = sort( perm2( t_perm));
            perm2_slice = non_perm2(t_perm(ind));

            tmp_corr = fast_corr( ...
                perm1_slice', ...
                perm2_slice');

            tcorr(slice_i, :, j, 1) = min(1);
            tcorr(slice_i, :, j, 2) = min(1);

        end
end
end



% perm = [ 32, 43, 154, 23 , 65, 13, 90, 1212, 12, ...]
% nonperm = [1,2,12,13,23,24,25,26,32,43,44,90,154]
% wins=[1,2,3],[2,3,4],[3,4,5],...
% out : [~,ind]=sort(perm(wins(i)); out = nonperm(ind);