function analyse_longitudinal_isc(ISC, alpha, kmax)
arguments
    ISC {is_corr(ISC)};
    alpha = 0.05;
    kmax = 10;
end
    addpath('utils/');
    ISC = checkfields(ISC);
    %isc_corr_mean_z = atanh(ISC.isc_corr_mean); % fisher transform

    fw_max = max(ISC.isc_corr_mean(2:end,:), [], 2);
    fw_min = min(ISC.isc_corr_mean(2:end,:), [], 2);
    r_crit_singlefeat_max = squeeze(quantile(fw_max, 1-(alpha/2), 1));
    r_crit_singlefeat_min = squeeze(quantile(fw_min, (alpha/2), 1));

    mean_subj_sign = ISC.isc_corr_mean(1, :);
    mean_subj_sign( mean_subj_sign < r_crit_singlefeat_max & ...
        mean_subj_sign > r_crit_singlefeat_min) = nan;

    plot_hist_h0nalt(reshape(ISC.isc_corr_mean(2:end,:), [],1), ...
        reshape(ISC.isc_corr_mean(1,:), [],1), ...
        alpha, r_crit_singlefeat_max, r_crit_singlefeat_min, ISC.outpath)
    
    % get the vertices indices for the kmax highest and lowest correlation values
    [values, indices] = maxk(mean_subj_sign(1, :, :), kmax, 2);
    [values_min, indices_min] = mink(abs(mean_subj_sign(1, :, :)), kmax, 2);

    indices = [indices, indices_min];
    values = [values, values_min];
    
    plot_res_and_save(ISC.mean_pos_3nn, values, indices, ...
        ISC.method, ISC.fps, ISC.prop_agreem, ISC.outpath, ISC.datapath, ...
        ISC.num_neigh, ISC.nb_features)
    
end

function is_corr(ISC)
    if ~matches(ISC.corr_method, 'corr')
        eid = 'Value:tcorr';
        msg = 'The ISC struct does not belong to a corr method';
        throwAsCaller(MException(eid, msg));
    end
end

function plot_hist_h0nalt(h0, alt, alpha, fw_max, fw_min, outpath)
    figure;
    histogram(alt,'Normalization','probability', 'BinWidth', 0.007);
    hold on; histogram(h0,'Normalization','probability', 'BinWidth', 0.007);
    line([fw_max fw_max],[0 0.1],...
    'color',[.2 .2 .2],...
    'linestyle', '--')
    line([fw_min fw_min],[0 0.1],...
    'color',[.2 .2 .2],...
    'linestyle', '--')
    hold off
    legend('data passing significance level', 'null distribution', 'fw 95perc', 'fw 5perc')
end