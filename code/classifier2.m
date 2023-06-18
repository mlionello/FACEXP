%% classifier
if false
pilotdata = load('../mediapipe/pilot/training_data.mat');
meddata = load('../mediapipe/ADFES_med/training_data.mat');
northdata = load('../mediapipe/ADFES_north/training_data.mat');
data.pilot = pilotdata.training_data; clear pilotdata;
data.med = meddata.training_data; clear meddata;
data.north = northdata.training_data; clear northdata;
end
%%

emo_id_ref = cellfun(@(x) str2num(x), {data.pilot.emo_id});
med_emo = string({data.med.emo_id});
north_emo =string({data.north.emo_id});
emo_dict = unique(med_emo);

north_emo(north_emo == "Anger") = 6;
north_emo(north_emo == "Surprise") = 8;
north_emo(north_emo == "Sadness") = 5;
north_emo(north_emo == "Joy") = 3;
north_emo(north_emo == "Fear") = 9;
north_emo(north_emo == "Disgust") = 10;
med_emo(med_emo == "Anger") = 6;
med_emo(med_emo == "Surprise") = 8;
med_emo(med_emo == "Sadness") = 5;
med_emo(med_emo == "Joy") = 3;
med_emo(med_emo == "Fear") = 9;
med_emo(med_emo == "Disgust") = 10;

rm_ind_north = [];
rm_ind_med = [];
for i = 1 : length(north_emo)
    if ~isempty(find(emo_dict == north_emo(i) ))
        rm_ind_north = [rm_ind_north, i];
    end
end
for i = 1 : length(med_emo)
    if ~isempty(find(emo_dict == med_emo(i) ))
        rm_ind_med = [rm_ind_med, i];
    end
end

features_med = cat(2, data.med.average_displacement)';
features_north = cat(2, data.north.average_displacement)';
features_med(rm_ind_med, :) = [];
features_north(rm_ind_north, :) = [];

north_emo(rm_ind_north) = [];
med_emo(rm_ind_med) = [];



%% TRAINING
rng(14051983)

features = cat(2,data.pilot.average_displacement)';
labels = cat(1, data.pilot.emo_id);
labels = string(str2num(char(labels)));

% labels(labels=="4") = 3;
% labels(labels=="6") = 7;

rm_tr_ind = [find(labels=="1"); find(labels=="2"); find(labels=="4"); find(labels=="7")];
%rm_tr_ind = [find(labels=="1"); find(labels=="2"); find(labels=="3"); find(labels=="6")];
%rm_tr_ind = [find(labels=="1"); find(labels=="2")];
features(rm_tr_ind, :) = [];
labels(rm_tr_ind, :) = [];

knn_model=fitcknn(features_north,...
    north_emo,...
    'NumNeighbors',5,...
    'DistanceWeight','inverse',...
    'BreakTies','nearest',...
    'Distance','euclidean');


labels_predicted = predict(knn_model,features);

Nconfusion_matrices = ...
    confusionmat(labels, labels_predicted);
acc_north = sum(diag(Nconfusion_matrices))/size(features,1)


knn_model=fitcknn(features_med,...
    med_emo,...
    'NumNeighbors',5,...
    'DistanceWeight','inverse',...
    'BreakTies','nearest',...
    'Distance','euclidean');

labels_predicted = predict(knn_model,features);

Mconfusion_matrices = ...
    confusionmat(labels,labels_predicted);
acc_med = sum(diag(Mconfusion_matrices))/size(features,1)

knn_model=fitcknn(cat(1,features_med, features_north),...
    cat(2, med_emo, north_emo),...
    'NumNeighbors',3,...
    'DistanceWeight','inverse',...
    'BreakTies','nearest',...
    'Distance','euclidean');

labels_predicted = predict(knn_model,features);

ALLconfusion_matrices = ...
    confusionmat(labels,labels_predicted);
acc_all = sum(diag(ALLconfusion_matrices))/size(features,1)
