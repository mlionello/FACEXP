##FACECIRCUS

#facecircus.m
facecircus(datapath) takes as argument the path where h5 data have been generate by ../vid2h5/get_h5_from_folder.py. 
fps: fps used in the video encoding;
nb_niegh: number of mutually exclusive neighbours for spatial smoothing;
method: either 'pdist' or 'l2'. in the first case the pairwise distance between each pair of the vertices output form the knn smoothing are calculated, in the latter case L@ norm is calcualted for each vertix;
corr_method: either 'tcorr' or 'corr': in the first case box of frames analysis is applied to extract stats for each window, in the second case the whole input data is seen by the statical analysis in once;
w_lens: window length expressed in frames. window lenght above which calculate an average in the case of 'corr' (corresponing to temporal smoothing), window box lenght for the punctual statistical analysis in case of 'tcorr'. In both cases a hop size of w_lens/3 is applied to the time series.


