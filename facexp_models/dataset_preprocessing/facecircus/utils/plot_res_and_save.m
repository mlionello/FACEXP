function plot_res_and_save(mean_pos_3nn, values, indices, w_lens, method, fps, prop_agreem, outpath, datapath, num_neigh, isc_corr)
indices= squeeze(indices);
values= squeeze(values);
template = h5read(datapath + '/h5out_30fps/local/l3celdsvq6yu4zufiag3rozjhmttv13n_output_local.h5', '/v');
template =  squeeze(mean(template, 3));
fig = figure('Visible', 0, 'Position',[0 0 1024 720]);
for win_i = 1: size(indices, 1)
    subplot(ceil(sqrt(length(w_lens))), ceil(sqrt(length(w_lens))), win_i)
for v =1: length(w_lens)
    cmap = jet;
    scatter(template(1, :), template(2, :));
    hold on;
    if matches(method, 'l2')
        for vx = 1:size(indices, 2)
            if isnan(values(win_i, vx))
                break
            end
            i = indices(win_i, vx);
            color_idx = round(interp1(linspace(-1, 1, size(cmap, 1)), 1:size(cmap, 1), values(win_i, vx)));
            if num_neigh>0
                scatter(mean_pos_3nn(1, i(1)), mean_pos_3nn(2, i(1)), 'MarkerFaceColor', cmap(color_idx, :))
            else
                scatter(template(1, i(1)), template(2, i(1)), 'MarkerFaceColor', cmap(color_idx, :))
            end
        end
    elseif matches(method, "pdist")
        for vx = 1: size(indices, 2)
            if isnan(values(win_i, vx))
                break
            end
            ind2sqr = zeros(1, size(isc_corr, 3));
            ind2sqr(indices(win_i, vx))=1;
            [i, j] = find(squareform(ind2sqr));
            color_idx = round(interp1(linspace(min(values,[],'all'), max(values,[],'all'), size(cmap, 1)), 1:size(cmap, 1), values(win_i, vx)));
            if num_neigh>0
                line([mean_pos_3nn(i(1), 1), mean_pos_3nn(j(1), 1)], [mean_pos_3nn(i(1), 2), mean_pos_3nn(j(1), 2)], 'Color', cmap(color_idx, :), 'LineWidth', 1 );
            else
                line([template(i(1), 1), template(j(1), 1)], [template(i(1), 2), template(j(1), 2)], 'Color', cmap(color_idx, :), 'LineWidth', 1);
            end
        end

    end
    title(compose("win length: %.2f s", w_lens(win_i)/fps))
    colormap(cmap);
    colorbar;
    hold off;
end
end
sgtitle(compose("agreement Proportion: %.1f", prop_agreem))
if num_neigh>0
    knn_fid = num_neigh;
else
    knn_fid = 0;
end
filename = compose("%s_win_%d_prop_%.1f_knn_%d.png", method, win_i, prop_agreem, knn_fid );
saveas(fig, fullfile(outpath,filename));
close(fig);

end
