%clc;

addpath(genpath('../libsvm-3.20'));

labelTrue = [ones(1,193), ones(1,200)*2];

for i_class = 1:2
    for i_cluster = 1:2
        a = find(labelTrue==i_class & clusterLabel ==i_cluster);
        lt = labelTrue(a);
        lp = labelpred(a);
        fprintf('class-%d, cluster-%d, acc:%.4f (total#:%d)\n', i_class, i_cluster, mean(lt==lp), length(lt));
    end
end

disp(squeeze(accTensor))

Y = round(labelTrue-1.5);
errorListTMP = errorList;
X = [errorListTMP; ones(1,size(errorList,2))];


alphaList = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10:3:120];
accList = [];
for i_alpha = 1:length(alphaList)
    alpha = alphaList(i_alpha);
    W = (X*X'+alpha*eye(size(X,1)))\X*Y';
    
    Yhat = W'*X;
    thr = 0.0;
    Yhat(Yhat<thr) = -1;
    Yhat(Yhat>=thr) = 1;
    %fprintf('alpha=%.4f, acc=%.4f\n', alpha, mean(Y==Yhat));
    accList(end+1) = mean(Y==Yhat);
end

[val, idx] = max(accList);
fprintf('linear reg acc: %.4f, (alpha=%.4f)\n', val, alphaList(idx));


%% svm

cl = fitcsvm(X', Y, ...
    'KernelFunction', 'linear',...
    'ClassNames', [-1,1]);

[Yhat, b] = predict(cl, X');
fprintf('linear SVM acc %.4f\n\n', mean(Yhat(:)==Y(:)));

