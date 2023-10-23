homepath = '/data1/EMOVIE_sampaolo/FACE/FaceCircus/data/h5out/';
load('/data1/EMOVIE_sampaolo/FACE/FaceCircus/data/FaceRatingsOverlap.mat')
indx_pp=1;
pp = find(FaceRatingsOverlap>15);
% ppnew = pp([1, find(pp(2:end)~=pp(1:end-1)+1)+1]);
% 
% vidsamples = cellfun(@string, {dir(fullfile(homepath, 'local')).name});
% 
% subj_id = FaceRatingsMetaData.SubjectsID(find(FaceRatings(ppnew(indx_pp)*10,:)==100),1);
% subj_id = cellfun(@string, subj_id);

%%
vertices_local1= h5read(fullfile(homepath, 'local/l4y76jn7hu88xia31gmbbpbfyutfk5ye_output-007_local.h5'), '/v'); 
vertices_local2= h5read(fullfile(homepath, 'local/8vfh5zszixyywwt8hcsaldts2xy6feda_output-003_local.h5'), '/v'); 
vertices_local3= h5read(fullfile(homepath, 'local/ag5s5gsvzpqwlvcpgp4yyhkrrzj4vpuh_output-008_local.h5'), '/v'); 
%vertices_raw = h5read(fullfile(homepath, 'raw/6zp74zbp0yrjtq7a3h2zwaukrtbj5y05_output-005_raw.h5'), '/v'); 
%%

figure();
fps = 60;
init_m = 10; init_s = 40; % minutes, seconds
init_s = (pp(indx_pp)-5); init_m=0;

L = 30*fps; % seconds*fps
init_frame = int32(init_m*60 + init_s + 1)*fps; % frames

while true
for i = init_frame:10:init_frame+L
subplot(1, 3, 1);
h1 = scatter(vertices_local1(1, :, i), vertices_local1(2, :, i), 'Or');
xlim([-10,10]); ylim([-10,10]);
subplot(1, 3, 2);
h2 = scatter(vertices_local2(1, :, i),vertices_local2(2, :, i), 'Or');
xlim([-10,10]); ylim([-10,10]);
subplot(1, 3, 3);
h3 = scatter(vertices_local3(1, :, i),vertices_local3(2, :, i), 'Or');
xlim([-10,10]); ylim([-10,10]);
title(num2str(floor(double(i)/60/60)) + ":" + num2str(floor((double(i)/60/60-floor(double(i)/60/60))*60)));
drawnow
end
end