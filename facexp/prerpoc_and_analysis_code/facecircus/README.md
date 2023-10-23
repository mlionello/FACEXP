# FACECIRCUS

## facecircus.m
facecircus(datapath) takes as argument the path where h5 data have been generate by ../vid2h5/get_h5_from_folder.py and it computes the intersubject correlation (isc) analysis.
For each subject it generate .mat according method and nb_neigh. These files will be loaded during the ISC analysis computed in utils/isc_pairwise_sampling.m and generate a ISC structure saved in a separate .mat file. the number of permutation is fixed to 200 in utils/isc_pairwise_sampling.m.
fps: fps used in the video encoding;
nb_niegh: number of mutually exclusive neighbours for spatial smoothing;
method: either 'pdist' or 'l2'. in the first case the pairwise distance between each pair of the vertices output form the knn smoothing are calculated, in the latter case L@ norm is calcualted for each vertix;
corr_method: either 'tcorr' or 'corr': in the first case box of frames analysis is applied to extract stats for each window, in the second case the whole input data is seen by the statical analysis in once;
w_lens: window length expressed in frames. window lenght above which calculate an average in the case of 'corr' (corresponing to temporal smoothing), window box lenght for the punctual statistical analysis in case of 'tcorr'. In both cases a hop size of w_lens/3 is applied to the time series.

# analyse_sliding_window_isc.m
analyse_longitudinal_isc(isc) it takes as argument the ISC structure genereated by facecircus.m when using a 'tcorr' method, and it generates a sequence of plots

# analyse_longitudinal_isc.m
analyse_longitudinal_isc(ISC, FaceRatingsProcessed, alpha, kmax, show_plot) it takes as argument the ISC structure genereated by facecircus.m when using a 'corr' method, and it generates a sequence of plots.
FaceRatingsProcessed is the path to the file containing the intensity reported by M subjects across M seconds of stimuli. It expects a matrix of sahpe 1 by N by M;
alpha (deafult = 0.05) is the significance level used against the null hypothesis;
kmax (deafult = 10) is both the number of top highest significant correlation the top lowest significant correlation surviving when tested for multiple comparisons.
show_plot (default = 0) whether to show the egenerated plots, or to save them.
