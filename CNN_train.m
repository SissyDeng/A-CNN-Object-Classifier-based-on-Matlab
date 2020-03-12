% Load the CIFAR-10 training and test data. 
trainingData=imageDatastore('image_train','IncludeSubfolders',true,'LabelSource','foldernames');
testdata=imageDatastore('image_test','IncludeSubfolders',true,'LabelSource','foldernames');

%建立自己的网络
 inputLayer = imageInputLayer([32 32 3]);
 filterSize = [5 5];
 numFilters = 32;

 middleLayers = [                
    convolution2dLayer(filterSize, numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride', 2)
    convolution2dLayer(filterSize, numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2)
    convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2) 
    ];
finalLayers = [
          fullyConnectedLayer(64)
          reluLayer()
          fullyConnectedLayer(10)
          softmaxLayer()
          classificationLayer()];
layers = [
    inputLayer
    middleLayers
    finalLayers
    ];
disp(layers);

opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);

doTraining = 1;
if doTraining    
    % Train a network.
    convnet = trainNetwork(trainingData,layers,opts);
else
    % Load pre-trained detector
    convnet = load('convnet.mat') ;      
end

w = convnet.Layers(2).Weights;
w = rescale(w);
figure
montage(w)

%测试网络
YTest = classify(cifar10Net,testdata);
TTest = testdata.Labels;
accuracy = sum(YTest == TTest)/numel(TTest);
disp(accuracy);