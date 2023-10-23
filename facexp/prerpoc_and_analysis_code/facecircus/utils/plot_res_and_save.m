function plot_res_and_save(mean_pos_3nn, values, indices, method, datapath, num_neigh, nb_features, outfig, show_plot)
mincmap = -0.2;
maxcmap = 0.3;
indices= squeeze(indices);
values= squeeze(values);
template = h5read(datapath + '/h5out_30fps/local/l3celdsvq6yu4zufiag3rozjhmttv13n_output_local.h5', '/v');
template =  squeeze(mean(template, 3));
if show_plot
    fig = figure('Visible', 1, 'Position',[0 0 1024 720]);
else
    fig = figure('Visible', 0, 'Position',[0 0 1024 720]);
end
if size(indices, 1) ~= 1
    disp("check size(indices, 1) ~= 1")
    return
end
for max_min_switch = 0:1
    init_vx = max_min_switch*size(indices, 2)/2+1;
    end_vx = (max_min_switch+1)*size(indices, 2)/2;
    subplot(1, 2, max_min_switch+1)
        if max_min_switch==0
            title(compose("max corr [%.2f, %.2f]",min(values(init_vx:end_vx)), max(values(init_vx:end_vx)) ))
        else
            title(compose("min abs corr [%.2f, %.2f]",min(values(init_vx:end_vx)), max(values(init_vx:end_vx)) ))
        end
    cmap = jet;
    scatter(template(1, :), template(2, :));
    hold on;
    for vx = init_vx: end_vx
        if isnan(values( vx))
            break
        end
        color_idx = 100;
        if length(values) > 1
            color_idx = round( ...
                interp1(linspace(min(mincmap,values(vx)), ...
                max(maxcmap,values(vx)), size(cmap, 1)), ...
                1:size(cmap, 1), values(vx)));
        end
        if matches(method, 'l2')
            i = indices(vx);
            if num_neigh>0
                if size(mean_pos_3nn, 2)==3
                    scatter(mean_pos_3nn(i(1), 1), mean_pos_3nn(i(1), 2), 'MarkerFaceColor', cmap(color_idx, :));
                else
                    scatter(mean_pos_3nn(1, i(1)), mean_pos_3nn(2, i(1)), 'MarkerFaceColor', cmap(color_idx, :))
                end
            else
                scatter(template(1, i(1)), template(2, i(1)), 'MarkerFaceColor', cmap(color_idx, :))
            end
        elseif matches(method, "pdist")
            ind2sqr = zeros(1, nb_features);
            ind2sqr(indices( vx)) = 1;
            [i, j] = find(squareform(ind2sqr));
            if num_neigh > 0
                line([mean_pos_3nn(i(1), 1), mean_pos_3nn(j(1), 1)], [mean_pos_3nn(i(1), 2), mean_pos_3nn(j(1), 2)], 'Color', cmap(color_idx, :), 'LineWidth', 1 );
            else
                line([template(i(1), 1), template(j(1), 1)], [template(i(1), 2), template(j(1), 2)], 'Color', cmap(color_idx, :), 'LineWidth', 1);
            end
        end
    end
    hold off;
    clim([mincmap, maxcmap]);
    colormap(cmap);
    colorbar;
end
if ~show_plot
    filename = compose("max_corr_significant_features.png" );
    saveas(fig, fullfile(outfig, filename));
    close(fig);
end
end
