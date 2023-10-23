function ISC = checkfields(ISC)
if ~isfield(ISC, 'fps')
    ISC.fps=30;
end
if ~isfield(ISC,'datapath')
    ISC.datapath = "/data1/EMOVIE_sampaolo/FACE/FaceCircus/data/";
end
if ~isfield(ISC, 'num_neigh')
    ISC.num_neigh = 5;
end 
if ~isfield(ISC, 'nb_features')
    ISC.nb_features = size(ISC.isc_corr_mean, 2);
end 
if ~isfield(ISC, 'nb_windows')
    ISC.nb_windows = size(ISC.isc_corr_mean, 3);
end 
if ~isfield(ISC, 'outpath')
    ISC.outpath = fullfile(ISC.datapath, ISC.method, sprintf("knn_%d", ISC.num_neigh));
end
end