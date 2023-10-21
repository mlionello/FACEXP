function analyse_longitudinal_isc(ISC, FaceRatingsProcessed, alpha, agreement_perc, kmax)
arguments
    ISC {is_corr(ISC)};
    FaceRatingsProcessed=[];
    alpha = 0.05;
    agreement_perc = 100;
    kmax = 10;
end
    addpath('utils/');
    ISC = checkfields(ISC);
    if isempty(FaceRatingsProcessed)
        FaceRatingsProcessed = load(fullfile(ISC.datapath, 'FaceRatingsProcessed.mat'));
        FaceRatingsProcessed = FaceRatingsProcessed.FaceRatingsProcessed;
    end
    isc_corr_mean_z = atanh(ISC.isc_corr_mean); % fisher transform

    % plot_hist_h0nalt(fw_max, isc_corr_mean(1, :), alpha, outpath)
    % r_crit_bf = quantile(fw_max, 1-(alpha/2));
    % [ind_feat, ind_wind] = find(isc_corr_mean(1, :, :) > repmat(r_crit, 1, size(isc_corr_mean, 2), 1));
    % isc_corr_mean(1, ind);

    fw_max = max(isc_corr_mean_z(2:end,:), [], 2);
    fw_min = min(isc_corr_mean_z(2:end,:), [], 2);
    r_crit_singlefeat_max = squeeze(quantile(fw_max, 1-(alpha/2), 1));
    r_crit_singlefeat_min = squeeze(quantile(fw_min, (alpha/2), 1));

    mean_subj_sign = isc_corr_mean_z(1, :);
    mean_subj_sign( mean_subj_sign<r_crit_singlefeat_max & ...
        mean_subj_sign>r_crit_singlefeat_min) = nan;

    plot_hist_h0nalt(reshape(isc_corr_mean_z(2:end,:), [],1), ...
        isc_corr_mean_z(1, :), ...
        alpha, r_crit_singlefeat_max, r_crit_singlefeat_min, ISC.outpath)
    
    % get the vertices indices for the 20 highest correlation values
    [values, indices] = maxk(mean_subj_sign(1, :, :), kmax, 2);
    %indices = find(mean_subj_sign(1, :, :) > 0.15);
    %values = mean_subj_sign(1, indices, :);
    
    plot_res_and_save(ISC.mean_pos_3nn, values, indices, ISC.w_lens, ...
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
    histogram(alt,100,'Normalization','probability');
    hold on; histogram(h0,15,'Normalization','probability');
    tcrit = quantile(h0,[alpha/2, 1-(alpha/2)]);
    line([fw_max fw_max],[0 0.1],...
    'color',[.2 .2 .2],...
    'linestyle', '--')
    line([fw_min fw_min],[0 0.1],...
    'color',[.2 .2 .2],...
    'linestyle', '--')
    hold off
end