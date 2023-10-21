function tcorr = compute_tcorr(nb_windows, nb_perm, nb_features, t_win, frame_indices, perms_sub1, sub1, sub2, perms_sub2)
%function [tcorr, fw_err, p_value] = compute_tcorr(nb_windows, nb_perm, nb_features, t_win, frame_indices, perms_sub1, sub1, sub2, perms_sub2)
hop_size = int32(t_win/3);
tcorr = cell(nb_windows, nb_perm);
frame_indices_slices_schema = get_frame_list_per_window(frame_indices, t_win, hop_size);
t_nonperm1 = perms_sub1(1, :);
t_nonperm2 = perms_sub2(1, :);

tcorr = cell(nb_windows, nb_perm);  % it will contain nb_feat by 1

parfor j = 1: nb_perm + 1
    t_perm1 =  perms_sub1(j, :);
    t_perm2 =  perms_sub2(j, :);
    for slice_i = 1 : nb_windows
            t_slice = frame_indices_slices_schema{slice_i};
        
            % I get the permutation order of the indices for the slice_i
            % window:
            [~, ind] = sort( t_perm1( t_slice));
            % the order of the permuted indices is used as random order 
            % for the indices in the slice_i non permutated window:
            t_perm1_star = t_nonperm1(t_slice(ind));
            % I apply the time indices order of the ith window to the
            % subject data:
            sub1_perm_slice = sub1(:, t_perm1_star);

            [~, ind] = sort( t_perm2( t_slice));
            t_perm2_star = t_nonperm2(t_slice(ind));
            sub2_perm_slice = sub2(:, t_perm2_star);

            tcorr{slice_i, j} = fast_corr( ...
                sub1_perm_slice', ...
                sub2_perm_slice')';

            % ok for each suj pair but not for mean_subj
            %if j == 1
            %    tcorr{slice_i, j} = tmp_corr;
            %    r_rank_comp = ones(size(tcorr{slice_i, 1}));
            %else
            %    r_rank_comp = r_rank_comp + tcorr{slice_i, 1} > tcorr{slice_i, j};
            %    fw_err{slice_i, j} = [...
            %        mean(tmp_corr), ...
            %        std(tmp_corr), ...
            %        max(tmp_corr), ...
            %        min(tmp_corr), ...
            %        quantile(tmp_corr, 0.95), ...
            %        quantile(tmp_corr, 0.05)];
            %end
    end
    % ok for each suj pair but not for mean_subj
    % p_value{slice_i} = 1 - r_rank_comp/(nb_perm+1);

end
end

% perm = [ 32, 43, 154, 23 , 65, 13, 90, 1212, 12, ...]
% nonperm = [1,2,12,13,23,24,25,26,32,43,44,90,154]
% wins=[1,2,3],[2,3,4],[3,4,5],...
% out : [~,ind]=sort(perm(wins(i)); out = nonperm(ind);