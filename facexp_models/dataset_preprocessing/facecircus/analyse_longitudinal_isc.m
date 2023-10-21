function analyse_longitudinal_isc(ISC, alpha, kmax, show_plot)
arguments
    ISC {is_corr(ISC)};
    alpha = 0.05;
    kmax = 10;
    show_plot = 0;
end
    addpath("utils/");
    ISC = checkfields(ISC);
    suffix = 1;
    outfig = fullfile(ISC.outpaht, 'figures', compose('kmax_%d_', kmax ));
    while exist(outfig+string(suffix), 'dir'); suffix = suffix + 1; end
    outfig = outfig + string(suffix);
    if ~show_plot; mkdir(outfig); end
    %isc_corr_mean_z = atanh(ISC.isc_corr_mean); % fisher transform

    % calculate corr coefs critical
    fw_max = max(ISC.isc_corr_mean(2:end,:), [], 2);
    fw_min = min(ISC.isc_corr_mean(2:end,:), [], 2);
    r_crit_singlefeat_max = squeeze(quantile(fw_max, 1-(alpha/2), 1));
    r_crit_singlefeat_min = squeeze(quantile(fw_min, (alpha/2), 1));
    % set to nan vertices in windows below significant level
    mean_subj_sign = squeeze(ISC.isc_corr_mean(1, :, :));
    mean_subj_sign( mean_subj_sign < r_crit_singlefeat_max' & ...
        mean_subj_sign > r_crit_singlefeat_min') = nan;
    % plot features distribution against null distribution
    plot_hist_h0nalt(reshape(ISC.isc_corr_mean(2:end,:, :), [],1), ...
        reshape(ISC.isc_corr_mean(1,:, :), [],1), ...
        r_crit_singlefeat_max, r_crit_singlefeat_min, outfig, show_plot)

    % get the vertices indices for the kmax highest and lowest correlation values
    [values, indices] = maxk(mean_subj_sign(1, :, :), kmax, 2);
    [values_min, indices_min] = mink(abs(mean_subj_sign(1, :, :)), kmax, 2);

    indices = [indices, indices_min];
    values = [values, values_min];
    
    plot_res_and_save(ISC.mean_pos_3nn, values, indices, ...
        ISC.method, ISC.datapath, ...
        ISC.num_neigh, ISC.nb_features, outfig, show_plot)

end

function is_corr(ISC)
    if ~matches(ISC.corr_method, 'corr')
        eid = 'Value:tcorr';
        msg = 'The ISC struct does not belong to a corr method';
        throwAsCaller(MException(eid, msg));
    end
end