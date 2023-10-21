function analyse_longitudinal_isc(ISC, FaceRatingsProcessed, alpha, kmax, show_plot)
arguments
    ISC {is_corr(ISC)};
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
    outfig = fullfile(ISC.outpath, 'figures', ...
        compose('kmax_%d_wlenFrames_%d_propagree_%dPerc_', ...
        kmax, ISC.w_lens, 100*ISC.prop_agreem));
    while exist(outfig+string(suffix), 'dir'); suffix = suffix + 1; end
    outfig = outfig + string(suffix);
    if ~show_plot; mkdir(outfig); end
    plot_frames_against_peak_agreement(FaceRatingsProcessed, show_plot, outfig, ISC.prop_agreem)
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
        reshape(ISC.isc_corr_mean(1,:, :), [], 1), ...
        r_crit_singlefeat_max, r_crit_singlefeat_min, outfig, show_plot)

    % get the vertices indices for the kmax highest and lowest correlation values
    [values, indices] = maxk(mean_subj_sign(1, :, :), kmax, 2);
    [values_min, indices_min] = mink(abs(mean_subj_sign(1, :, :)), kmax, 2);

    indices = [indices, indices_min];
    values = [values, values_min];
    
    plot_res_and_save(ISC.mean_pos_3nn, values, indices, ...
        ISC.method, ISC.datapath, ...
        ISC.num_neigh, ISC.nb_features, outfig, show_plot)


%     if ISC.prop_agreem>0
%         % SELECT ONLY THE FRAMES WEHERE AT LEAST A CERTAIN NUMBER OF PARTICIPANTS
%         % AGREE ON A PEAK
%         t_indices = find(sum(squeeze(FaceRatingsProcessed(1, :, :)==100), 2) > ISC.n_subjs*ISC.prop_agreem);
%         frame_indices = cellfun(@(t) (t-2)*ISC.fps : (t+2)*ISC.fps, num2cell(t_indices), ...
%             'UniformOutput', false);
%         frame_indices = unique(cell2mat(frame_indices))';
%     else 
%         frame_indices = 60: size(FaceRatingsProcessed, 2)*fps;
%     end
%     load(fullfile(ISC.outpath, 'subjects', 'metadata'));
%     metadata.timepoints(2) = [];
%     frame_indices = frame_indices(frame_indices <= min(metadata.timepoints));
%     %%%% CHECK SUBJ 2!!!!!!!!!!!!!!!
% 
%     in_folder = fullfile(ISC.outpath, 'subjects');
%     h5_files = {dir(fullfile(in_folder, '*_output_local.mat')).name};
%     for j = 1:length(h5_files)
%         if j==2
%             continue
%         end
%         data_file_path = fullfile(ISC.outpath, 'subjects', h5_files(j));
%         data = load( data_file_path);
%         data= data.indata;
%         if j==1
%             mean_subj =  data(:,frame_indices)/(length(ISC.subj_id)-1);
%         else
%             mean_subj = mean_subj + data(:,frame_indices)/(length(ISC.subj_id)-1);
%         end
%     end
%     behav = sum(FaceRatingsProcessed == 100, 3);
%     oL = length(behav);
%     downsampled_behav = interp1(1:oL, behav, linspace(1, oL, size(mean_subj, 2)));
%     for vx =1:size(mean_subj,1)
%         behav_feat_corr(vx) =  corr(downsampled_behav', mean_subj(vx,:)');
%     end
% 
%     figure; plot(mean_subj(indices(1),:)); hold on; plot(mean_subj(indices(1),:)); 

end

function plot_frames_against_peak_agreement(FaceRatingsProcessed, show_plot, outfig, prop_agreem)
    if show_plot
        fig = figure('Visible', 1, 'Position',[0 0 1024 400]);
    else
        fig = figure('Visible', 0, 'Position',[0 0 1024 400]);
    end
    behav = sum(squeeze(FaceRatingsProcessed(1, :, :)==100), 2);
    plot(behav)
    xlabel('seconds')
    yline(21*0.4, '--')
    ylabel('number of participants agreeing on a peak')
    hold on;
    yline(21*prop_agreem, '--')
    x = behav; x(x<21*prop_agreem)=21*prop_agreem;  fill([1:length(behav), length(behav):-1:1], [x; 21*prop_agreem*ones(size(x))]', 'red', 'FaceAlpha', 0.2)
    title('frame selections according to percentage of agreement on intensity peaks')
    hold off;
    if ~show_plot
        filename = compose("max_corr_significant_features_specific.png" );
        saveas(fig, fullfile(outfig, filename));
        close(fig);
    end
    
    if show_plot
        fig = figure('Visible', 1, 'Position',[0 0 1024 400]);
    else
        fig = figure('Visible', 0, 'Position',[0 0 1024 400]);
    end
    behav = sum(squeeze(FaceRatingsProcessed(1, :, :)==100), 2);
    plot(behav)
    xlabel('seconds')
    ylabel('number of participants agreeing on a peak')
    hold on;
    yline(21*0.4, '--')
    yline(21*0.8, '--')
    yline(21*0.6, '--')
    x = behav; x(x<21*0.6)=21*0.6;  fill([1:length(behav), length(behav):-1:1], [x; 21*0.6*ones(size(x))]', 'red', 'FaceAlpha', 0.2)
    x = behav; x(x<21*0.8)=21*0.8;  fill([1:length(behav), length(behav):-1:1], [x; 21*0.8*ones(size(x))]', 'black', 'FaceAlpha', 0.9)
    x = behav; x(x<21*0.4)=21*0.4;  fill([1:length(behav), length(behav):-1:1], [x; 21*0.4*ones(size(x))]', 'green', 'FaceAlpha', 0.1)
    title('frame selections according to percentage of agreement on intensity peaks')
    hold off;
    if ~show_plot
        filename = compose("max_corr_significant_features_general.png" );
        saveas(fig, fullfile(outfig, filename));
        close(fig);
    end
end

function is_corr(ISC)
    if ~matches(ISC.corr_method, 'corr')
        eid = 'Value:tcorr';
        msg = 'The ISC struct does not belong to a corr method';
        throwAsCaller(MException(eid, msg));
    end
end