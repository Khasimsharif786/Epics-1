clc;
close all;
%net=nasnetlarge();
%layers=net.Layers;
a=1;
m=224;
n=224;
imds=datastore("microbes","IncludeSubfolders",true,"LabelSource","foldernames");
numClasses = numel(categories(imds.Labels));
[trainds,testds]=splitEachLabel(imds,0.8);
ltrain=trainds.Labels;
ltest=testds.Labels;
trainds=augmentedImageDatastore([m n],trainds,"ColorPreprocessing","gray2rgb");
testds=augmentedImageDatastore([m n],testds,"ColorPreprocessing","gray2rgb");
auds=augmentedImageDatastore([m n],imds,"ColorPreprocessing","gray2rgb");
if input("to train network for new data press 1 and enter:")==1
    y=input("press 1 for sgdm,2 for adam,3 for rmsprop ");
    alg=["sgdm" "adam" "rmsprop"];
    figure(a)
    opts = trainingOptions(alg(y),"InitialLearnRate",0.001,"MaxEpochs",150,"VerboseFrequency",1,'Plots','training-progress');
    net1 = trainNetwork(trainds,new(numClasses),opts);
    a=a+1;
    figure(a)
    opts = trainingOptions(alg(y),"InitialLearnRate",0.001,"MaxEpochs",150,"VerboseFrequency",1,'Plots','training-progress');
    net2 = trainNetwork(trainds,new2(numClasses),opts);
    a=a+1;
    figure(a)
    opts = trainingOptions(alg(y),"InitialLearnRate",0.001,"MaxEpochs",150,"VerboseFrequency",1,'Plots','training-progress');
    net3 = trainNetwork(trainds,new3(numClasses),opts);
    a=a+1;
end
%plot(info.TrainingLoss);
fname=imds.Files;
name=input("enter img no to predict:");
img=readimage(imds,name);
disp("image true class:")
disp(fname(name))
% img=imread("Acinetobacter.baumanii,Actinomyces.israeli,Bacteroides.fragilis,Bifidobacterium.spp.jpeg");
if length(size(img))==2
    img=cat(3,img,img,img);
    %rgbImage = ind2rgb(grayImage, colormap);
end
img=imresize(img,[m n]);
disp("Googlenet prediction and test accuracy:")
a=predections(img,net1,testds,ltest,a,"Googlenet");
disp("Resnet-50 prediction and test accuracy:")
a=predections(img,net2,testds,ltest,a,"Resnet-50");
disp("Efficientnet-b0 prediction and test accuracy:")
a=predections(img,net3,testds,ltest,a,"Efficientnet-b0");
function a=predections(img,net,auds,labels,a,netname)
%preds=classify(net,trainds);
preds1=classify(net,auds);
Actual=labels;
numcorrect=nnz(Actual==preds1);
fraccorrect=numcorrect/numel(preds1);
disp(fraccorrect*100)
figure(a)
% subplot(1,2,1)
confusionchart(Actual,preds1)
title(netname+" Confusion matrix")
conf_matrix = confusionmat(Actual, preds1);
numClasses = size(conf_matrix, 1);  % Get the number of classes
disp("Precision,Recall & F1 Score respectively:")
precisions = zeros(numClasses, 1);
recalls = zeros(numClasses, 1);
f1Scores = zeros(numClasses, 1);

for i = 1:numClasses
    TP = conf_matrix(i, i);  % True Positives for class i
    FP = sum(conf_matrix(i, :)) - TP;  % False Positives for class i
    FN = sum(conf_matrix(:, i)) - TP;  % False Negatives for class i

    precisions(i) = TP / (TP + FP);
    recalls(i) = TP / (TP + FN);
    f1Scores(i) = 2 * (precisions(i) * recalls(i)) / (precisions(i) + recalls(i));
end

% Display or use the metrics for each class
disp(mean(precisions))
disp(mean(recalls))
for i=1:length(f1Scores)
if isnan(f1Scores(i))
    f1Scores(i)=0;
end
end
disp(mean(f1Scores))

a=a+1;
% idxWrong = find(preds1 ~= Actual);
% idx = idxWrong(1)
% if idx>0
%     figure(a)
%     imshow(readimage(auds,idx))
%     title(imds.Labels(idx))
%     a=a+1;
% end
[pred,scores]=classify(net,img);
layer=net.Layers;
inlayer=layer(1);
outlayer=layer(end);
insize=inlayer.InputSize;
category=outlayer.Classes;
%bar(scores);
highscores=scores>0.01;
% subplot(1,2,2)
figure(a)
bar(scores(highscores));
xlabel("Classes")
ylabel("Scores")
title(netname+" Prediction scores")
a=a+1;
xticklabels(category(highscores));
disp(pred)
figure(a)
imshow(img)
title(netname+" Prediction: "+string(pred))
a=a+1;
end