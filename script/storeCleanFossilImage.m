% save fossilPollenRecord.mat
clear
clc
close all;

%% load log
load fossilPollenRecord.mat;

idx = find(conf>95);

idx_Pos = Position(idx);
idx_Name = PietraID(idx);
idx_Slide = Slide(idx);

%% fetch clean fossil images
datasetDIR = '../database';


for i = 1:length(idx)
    name = idx_Name{i};
    pos = idx_Pos(i);
    slide = idx_Slide{i};
    
    namepath = fullfile( datasetDIR, strcat(name, ' fossil'), strcat(slide, [' ' 'Pos'], [' ' num2str(pos)], '.jpg') );
    
    C = strsplit(slide,' ');
    strC = [];
    for j = 1:numel(C)
        strC = ['*', C{j}];
    end
    a = dir( fullfile(datasetDIR, strcat(name, ' fossil'), strcat(strC, ['*', 'Pos', ' ', num2str(pos)], '.jpg') ) );
    
    if i == 41
        fprintf('id-%d, %d, exist!\n', i, length(a));
    else
        fprintf('id-%d, %d, none!\n', i, length(a));
    end
end



