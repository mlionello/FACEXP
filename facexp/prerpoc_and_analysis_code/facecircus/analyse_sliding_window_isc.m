function analyse_sliding_window_isc(ISC, FaceRatingsProcessed, alpha, kmax, show_plot)
arguments
    ISC {is_tcorr(ISC)};
    FaceRatingsProcessed = [];
    alpha = 0.05;
    kmax = 10;
    show_plot = 0;
end
    addpath("utils/");
    ISC = checkfields(ISC);
    if isempty(FaceRatingsProcessed)
        FaceRatingsProcessed = load(fullfile(ISC.datapath, 'FaceRatingsProcessed.mat'));
        FaceRatingsProcessed = FaceRatingsProcessed.FaceRatingsProcessed;
    end
    suffix = 1;
    outfig = fullfile(ISC.outpath, 'figures', compose('kmax_%d_%s_winlen_%d_', kmax, ISC.corr_method, ISC.w_lens));
    while exist(outfig+string(suffix), 'dir'); suffix = suffix + 1; end
    outfig = outfig + string(suffix);
    if ~show_plot; mkdir(outfig); end
    %isc_corr_mean_z = atanh(ISC.isc_corr_mean); % fisher transform
    
    % calculate corr coefs critical
    fw_max = max(ISC.isc_corr_mean(2: end, :, :), [], 2);
    fw_min = min(ISC.isc_corr_mean(2: end, :, :), [], 2);
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

    % downsampling behavioural cumulative agreements by interpolation
    beahv = sum(FaceRatingsProcessed == 100, 3);
    %beahvd = FaceRatingsOverlap;        
    oL = length(beahv);
    downsampled_behav = interp1(1:oL, beahv, linspace(1, oL, size(fw_max, 3)));
    
    % extract correlation and pvalues for window correlation series against behaviourals for each features
    for feat = 1: ISC.nb_features
        [corr_beahv(feat), pvalue_corr(feat)] = corr(downsampled_behav', ...
            squeeze(ISC.isc_corr_mean(1, feat, :)));
    end
    
    % get the vertices indices for the kmax highest and lowest correlation
    % values between pdist and behavioral data, below significant level
    corr_beahv_sign = corr_beahv;
    corr_beahv_sign(pvalue_corr > alpha/2) = nan;
    [~, corr_feat_ind_max] = maxk(abs(corr_beahv_sign), kmax);
    corr_feat_value_max = corr_beahv_sign(corr_feat_ind_max);
    [~, corr_feat_ind_min] = mink(abs(corr_beahv_sign), kmax);
    corr_feat_value_min = corr_beahv_sign(corr_feat_ind_min);
    corr_feat_value = [corr_feat_value_max, corr_feat_value_min];
    corr_feat_ind = [corr_feat_ind_max, corr_feat_ind_min];

    % plot corr temporal series for the features with top significant correlation with behav.
    plot_tseries(downsampled_behav, ISC, corr_feat_ind_max, corr_feat_value_max, r_crit_singlefeat_min, r_crit_singlefeat_max, outfig, show_plot)
    % same but with facial space representation of the features
    plot_tseries_and_face(downsampled_behav, ISC, corr_feat_ind_max, corr_feat_value_max, r_crit_singlefeat_min, r_crit_singlefeat_max, outfig, show_plot)
    
    % scatter facial pdist with highest and absolute lowest correlation
    % from the kmax features with highest corr with behavioural data
    plot_res_and_save(ISC.mean_pos_3nn, corr_feat_value, ...
        corr_feat_ind, ISC.method, ISC.datapath, ISC.num_neigh, ...
        ISC.nb_features, outfig, show_plot);
end

function is_tcorr(ISC)
    if ~matches(ISC.corr_method, 'tcorr')
        eid = 'Value:tcorr';
        msg = 'The ISC struct does not belong to a tcorr method';
        throwAsCaller(MException(eid, msg));
    end
end

function plot_tseries(downsampled_behav, ISC,corr_feat_ind_max, corr_feat_value_max, r_crit_singlefeat_min, r_crit_singlefeat_max, outfig, show)
    if show
        fig = figure('Visible','on', 'Position',[0 0 2000 1000]);
    else
        fig = figure('Visible','off', 'Position',[0 0 2000 1000]);
    end
    for i = 1:length(corr_feat_ind_max)
        downsampled_behav_norm = downsampled_behav/max(downsampled_behav);
        downsampled_behav_norm = downsampled_behav_norm*max(ISC.isc_corr_mean(1, corr_feat_ind_max(i), :));
        subplot(ceil(length(corr_feat_ind_max)/3), 3, i);
        plot(squeeze(ISC.isc_corr_mean(1, corr_feat_ind_max(i), :))); hold on;
        plot(squeeze(downsampled_behav_norm), 'r');
        plot(r_crit_singlefeat_max);
        plot(r_crit_singlefeat_min);
        fill([1 : ISC.nb_windows, ISC.nb_windows: -1: 1],...
            [r_crit_singlefeat_max', fliplr(r_crit_singlefeat_min')], ...
             'r', 'FaceAlpha', 0.3);
        hold off;
        %legend({'isc', 'behav', 'max fw 95', 'min fw 05'});
        ind2sqr = zeros(1, ISC.nb_features);
        ind2sqr(corr_feat_ind_max(i)) = 1;
        [kk, jj] = find(squareform(ind2sqr));
        title(compose('feat %d [%d, %d] with corr btw behav and isc: %.2f;', corr_feat_ind_max(i), kk(1), jj(1), corr_feat_value_max(i)))
    end
    if ~show
        saveas(fig, fullfile(outfig, 'time_series_against_ref_and_null_for_topcorr_features.png'));
        close(fig)
    end
end

function plot_tseries_and_face(downsampled_behav, ISC, corr_feat_ind_max, corr_feat_value_max, r_crit_singlefeat_min, r_crit_singlefeat_max, outfig, show)
    template = h5read(ISC.datapath + '/h5out_30fps/local/l3celdsvq6yu4zufiag3rozjhmttv13n_output_local.h5', '/v');
    template =  squeeze(mean(template, 3));
    for i = 1:length(corr_feat_ind_max)
        if show
            fig = figure('Visible','on', 'Position',[0 0 2000 300]);
        else
            fig = figure('Visible','off', 'Position',[0 0 2000 300]);
        end
        subplot(1,5,1:4);
        downsampled_behav_norm = downsampled_behav/max(downsampled_behav);
        downsampled_behav_norm = downsampled_behav_norm*max(ISC.isc_corr_mean(1, corr_feat_ind_max(i), :));
        plot(squeeze(ISC.isc_corr_mean(1, corr_feat_ind_max(i), :))); hold on;
        plot(squeeze(downsampled_behav_norm), 'r');
        plot(r_crit_singlefeat_max);
        plot(r_crit_singlefeat_min);
        fill([1 : ISC.nb_windows, ISC.nb_windows: -1: 1],...
            [r_crit_singlefeat_max', fliplr(r_crit_singlefeat_min')], ...
             'r', 'FaceAlpha', 0.3);
        hold off;
        %legend({'isc', 'behav', 'max fw 95', 'min fw 05'});
        if matches(ISC.method, "pdist")
            ind2sqr = zeros(1, ISC.nb_features);
            ind2sqr(corr_feat_ind_max(i)) = 1;
            [kk, jj] = find(squareform(ind2sqr));
            title(compose('feat %d [%d, %d] with corr btw behav and isc: %.2f;', corr_feat_ind_max(i), kk(1), jj(1), corr_feat_value_max(i)))
        elseif matches(ISC.method, "l2")
            kk = corr_feat_ind_max(i);
            title(compose('feat %d with corr btw behav and isc: %.2f;', kk, corr_feat_value_max(i)))
        end
        subplot(155)
        scatter(template(1, :), template(2, :));
        hold on;
        if ISC.num_neigh > 0 && matches(ISC.method, "pdist")
            line([ISC.mean_pos_3nn(kk(1), 1), ISC.mean_pos_3nn(jj(1), 1)], ...
                [ISC.mean_pos_3nn(kk(1), 2), ISC.mean_pos_3nn(jj(1), 2)], ...
                'Color', 'black', 'LineWidth', 1 );
        elseif ISC.num_neigh == 0 && matches(ISC.method, "pdist")
            line([template(kk(1), 1), template(jj(1), 1)], ...
                [template(kk(1), 2), template(jj(1), 2)], ...
                'Color', 'black', 'LineWidth', 1);
        elseif ISC.num_neigh>0 && matches(ISC.method, "l2")
            if size(ISC.mean_pos_3nn, 2)==3
                scatter(ISC.mean_pos_3nn(kk, 1), ISC.mean_pos_3nn(kk, 2), 'k', 'filled');
            else
                scatter(ISC.mean_pos_3nn(1, kk), ISC.mean_pos_3nn(2, kk), 'k', 'filled')
            end
        elseif ISC.num_neigh==0 && matches(ISC.method, "l2")
            scatter(template(1, kk), template(2, kk), 'k', 'filled')
        end
        hold off;
        if ~show
            if matches(ISC.method, "pdist")
                filename2save=compose('feat_%d_pair_%d_%d_corr_%.2f.png', corr_feat_ind_max(i), kk(1), jj(1), corr_feat_value_max(i));
            elseif matches(ISC.method, "l2")
                filename2save=compose('feat_%d_corr_%.2f.png', kk, corr_feat_value_max(i));
            end
            saveas(fig, fullfile(outfig, filename2save));
            close(fig)
        end
    end
end
