clear
close all
clc

%% load result
load slideInfo.mat
load imNameList.mat

glaucaName = imNameList{1};
marianaName = imNameList{2};


glaucaErrorList = errorList(:,length(marianaName)+1:end);
marianaErrorList = errorList(:,1:length(marianaName));

glaucaTrueLabel = labelTrue(length(marianaName)+1:end);
marianaTrueLabel = labelTrue(1:length(marianaName));

glaucaPredLabel = labelpred(length(marianaName)+1:end);
marianaPredLabel = labelpred(1:length(marianaName));

% check
a = double(glaucaErrorList(1,:)<glaucaErrorList(2,:));
a(find(a==0))=2;
disp(sum(double(a~=glaucaPredLabel)))



%% Pietra record as hash table
PietraConf = containers.Map;
for i = 1:length(PietraID)
    slideTMP = Slide{i};
    slideTMP = strsplit(slideTMP);
    slideTMP = slideTMP{length(slideTMP)};
    key = strcat(PietraID{i}, '_', slideTMP, '_', Position{i});
    if isKey(PietraConf, key)
        fprintf('\t%s is already in the hash table!\n', key);
    end
    PietraConf(key) = PietraConfidence(i);
end

%keys(PietraConf)
%values(PietraConf)

%% glauca
%%{
fprintf('\n\nglauca...\n')
badImageCount=0;
GlaucaStruct = struct;

anchor = 1;
for gIdx = 2:length(glaucaName)
    name = glaucaName(gIdx).name;
    if strcmp(name, 'Nelson Lake NE830 Pos 105 - Grain 1_K1.mat')
        
    else
        C1 = strsplit(name, 'Pos');
        C2 = strtrim(C1{1});
        C2 = strsplit(C2);
        C3 = strsplit(strtrim(C1{2}),'_');
        C3 = strsplit(C3{1});
        PositionName = C3{1};
        slideName = C2{length(C2)};
        %fprintf('%s --------slide:%s---------position:%s---\n', name, slideName, PositionName);
        
        if strcmp(slideName, '1010')
            slideName = strcat('NE',slideName);
        end
        
        GlaucaStruct(anchor).trueLabel = glaucaTrueLabel(gIdx);
        GlaucaStruct(anchor).predLabel = glaucaPredLabel(gIdx);
        GlaucaStruct(anchor).slide = slideName;
        GlaucaStruct(anchor).imageName = name;
        GlaucaStruct(anchor).position = PositionName;
        GlaucaStruct(anchor).reconError = glaucaErrorList(:,gIdx); %1st row is the error by glauca, 2nd one is by mariana
        key = strcat('glauca_', slideName, '_', PositionName);
        
        
        
        if strcmp(name, 'Nelson Lake NE830 Pos 105 - Grain 2_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE830_105-2';
        elseif strcmp(name, 'Nelson Lake NE830 Pos 26 - Grain 1_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE830_26-1';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 198 Grain 1 cropped_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_198-1';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 198 Grain 3 cropped_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_198-3';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 238 Grain 1 cropped_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_238-1';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 238 Grain 2 cropped_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_238-2';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 247 bottom cropped_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_247-1';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 247 top cropped_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_247-2';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 294 left cropped_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_294-1';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 294 right cropped_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_294-2';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 297 bottom cropped - BUMP at slice 28_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_297-2';
        elseif strcmp(name, 'Nelson Lake NE978 Pos 297 top cropped - BUMP at slice 28_K1.mat')
            GlaucaStruct(anchor).key = 'glauca_NE978_297-1';
        else
            GlaucaStruct(anchor).key = key;
        end
        
        GlaucaStruct(anchor).confidence = PietraConf(GlaucaStruct(anchor).key);
        
        if ~isKey(PietraConf, GlaucaStruct(anchor).key)
            badImageCount = badImageCount + 1;
            fprintf('\t%s--slide:%s--position:%s--\n', name, slideName, PositionName);
            %fprintf('\t%s\n', key);
        end
        anchor = anchor + 1;
    end
end

fprintf('%d images are found in glauca!\n', length(GlaucaStruct));
fprintf('%d unmatched images are found in glauca!\n', badImageCount);
%}


%% mariana
%%{
fprintf('\n\nmariana...\n')
badImageCount=0;
MarianaStruct = struct;
anchor = 1;
for gIdx = 1:length(marianaName)
    name = marianaName(gIdx).name;
    C1 = strsplit(name, 'Pos');
    C2 = strtrim(C1{1});
    C2 = strsplit(C2);
    C3 = strsplit(strtrim(C1{2}),'_');
    C3 = strsplit(C3{1});
    PositionName = C3{1};
    slideName = C2{length(C2)};
    %fprintf('%s --------slide:%s---------position:%s---\n', name, slideName, PositionName);
    
    if strcmp(slideName, '1010')
        slideName = strcat('NE',slideName);
    end
    
    MarianaStruct(anchor).trueLabel = marianaTrueLabel(gIdx);
    MarianaStruct(anchor).predLabel = marianaPredLabel(gIdx);
    MarianaStruct(anchor).slide = slideName;
    MarianaStruct(anchor).imageName = name;
    MarianaStruct(anchor).position = PositionName;
    MarianaStruct(anchor).reconError = marianaErrorList(:,gIdx); %1st row is the error by glauca, 2nd one is by mariana
    
    key = strcat('mariana_', slideName, '_', PositionName);
    
    if strcmp(name, 'Nelson Lake NE830 Pos 26 - Grain 2_K1.mat')
        MarianaStruct(anchor).key = 'mariana_NE830_26-2';
    elseif strcmp(name, 'Nelson Lake NE978 Pos 198 Grain 2 cropped_K1.mat')
        MarianaStruct(anchor).key = 'mariana_NE978_198-2';
    elseif strcmp(name, 'Nelson Lake NE978 Pos 327 bottom cropped_K1.mat')
        MarianaStruct(anchor).key = 'mariana_NE978_327-1';
    elseif strcmp(name, 'Nelson Lake NE978 Pos 327 top cropped_K2.mat')
        MarianaStruct(anchor).key = 'mariana_NE978_327-2';
    else
        MarianaStruct(anchor).key = key;
    end
    MarianaStruct(anchor).confidence = PietraConf(MarianaStruct(anchor).key);
    
    %if ~isKey(PietraConf, key) %
    if ~isKey(PietraConf, MarianaStruct(anchor).key)
        badImageCount = badImageCount + 1;
        fprintf('\t%s--slide:%s--position:%s--\n', name, slideName, PositionName);
        %fprintf('\t%s\n', key);
    end
    
    anchor = anchor + 1;
end
fprintf('%d images are found in mariana!\n', length(MarianaStruct));
fprintf('%d unmatched images are found in mariana!\n', badImageCount);



%% output the detailed results
fprintf('image name, Slide, Position, PietraID, Pietra Confidence, true label (same as PietraID), predicted label, reconstruction error by glauca, reconstruction error by mariana\n');

labtrue = [];
labpred = [];
conf = [];
for anchor = 1:length(GlaucaStruct)
    fprintf('%s, ',    GlaucaStruct(anchor).imageName);
    fprintf('%s, ',    GlaucaStruct(anchor).slide);
    fprintf('%s, glauca, ',    GlaucaStruct(anchor).position);
    fprintf('%d, ',    GlaucaStruct(anchor).confidence);
    fprintf('%d, ',    GlaucaStruct(anchor).trueLabel);
    fprintf('%d, ',    GlaucaStruct(anchor).predLabel);
    fprintf('%.4f, ',  GlaucaStruct(anchor).reconError(1));
    fprintf('%.4f \n',  GlaucaStruct(anchor).reconError(2));
    labtrue = [labtrue, GlaucaStruct(anchor).trueLabel];
    labpred = [labpred, GlaucaStruct(anchor).predLabel];
    conf = [conf, GlaucaStruct(anchor).confidence];
end
for anchor = 1:length(MarianaStruct)
    fprintf('%s, ',    MarianaStruct(anchor).imageName);
    fprintf('%s, ',    MarianaStruct(anchor).slide);
    fprintf('%s, mariana, ',    MarianaStruct(anchor).position);
    fprintf('%d, ',    MarianaStruct(anchor).confidence);
    fprintf('%d, ',    MarianaStruct(anchor).trueLabel);
    fprintf('%d, ',    MarianaStruct(anchor).predLabel);
    fprintf('%.4f, ',  MarianaStruct(anchor).reconError(1));
    fprintf('%.4f \n',  MarianaStruct(anchor).reconError(2));
    labtrue = [labtrue, MarianaStruct(anchor).trueLabel];
    labpred = [labpred, MarianaStruct(anchor).predLabel];
    conf = [conf, MarianaStruct(anchor).confidence];
end

%% accuracy with varying confidence
acc_conf = [];
for confThresh = 0:1:99
    idx = find(conf>=confThresh);
    acc = mean(labpred(idx)==labtrue(idx));
    fprintf('accuracy:%.4f,\t%d\n', acc, confThresh);
    acc_conf = [acc_conf, acc];
end

figure;
plot(0:99, acc_conf, '.-r');
xlabel('confidence');
ylabel('accuracy');
title('accuracy vs. confidence')


