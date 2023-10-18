function frame_indices_slices = get_frame_list_per_window(frame_indices, t_win, hop_size)
        timepoints = length(frame_indices);
        frame_indices_slices = cell(floor((timepoints - t_win) / hop_size), 1);
        
        for t = 1: floor((timepoints - t_win) / hop_size)
            init_t =  (t - 1) * hop_size + 1;
            frame_indices_slices{t} = init_t : init_t + t_win;
        end
end