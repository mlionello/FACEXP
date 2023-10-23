function plot_hist_h0nalt(h0, alt, fw_max, fw_min, outpath, show)
    if show
        figure('Visible', 1)
    else
        fig = figure('Visible', 0);
    end
    hold on;
    histogram(alt,'Normalization','probability', 'BinWidth', 0.0077);
    histogram(h0,'Normalization','probability', 'BinWidth', 0.007);
    line([fw_max fw_max],[0 0.1],...
    'color',[.2 .2 .2],...
    'linestyle', '--')
    line([fw_min fw_min],[0 0.1],...
    'color',[.2 .2 .2],...
    'linestyle', '--')
    hold off
    legend('data passing significance level', 'null distribution', 'fw 95perc', 'fw 5perc')

    if ~show
        saveas(fig, fullfile(outpath, 'hist_features_against_null.png'));
        close(fig)
    end
end