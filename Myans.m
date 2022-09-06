imds=imageDatastore("WormImages");
groundtruth=readtable("WormData.csv");
imds.Labels=categorical(groundtruth.Status);
[imdtrain, imdtest]=splitEachLabel(imds,0.6,"randomized");
atrain=augmentedImageDatastore([224 224],imdtrain,"ColorPreprocessing","gray2rgb");
atest=augmentedImageDatastore([224 224],imdtest,"ColorPreprocessing","gray2rgb");
%%
options=trainingOptions("sgdm","InitialLearnRate",0.001,ExecutionEnvironment="multi-gpu");
[Mynet,info]=trainNetwork(atest,lgraph_1,options);
testpreds=classify(Mynet,atest);
%%
reality=imdtest.Labels;
nnz(testpreds==reality)/numel(testpreds)

confusionchart(reality,testpreds)
%%
idx = find(testpreds~=reality)
if ~isempty(idx)
    imshow(readimage(imdtest,idx(1)))
    title(reality(idx(1)))
end