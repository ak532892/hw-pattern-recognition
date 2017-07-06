clc
clearvars
load mnist_all.mat

dataRow = [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949 ...
            980 1135 1032 1010 982 892 958 1028 974 1009];
dataString = {'train0' 'train1' 'train2' 'train3' 'train4' 'train5' ...
              'train6' 'train7' 'train8' 'train9' ...
              'test0' 'test1' 'test2' 'test3' 'test4' 'test5' 'test6' ...
              'test7' 'test8' 'test9'};
          
% Merge
trainDataCount = 1000;
trainImage = train0(1:trainDataCount, 1:end);
% Label = 1~10 for convinence
VT(1:trainDataCount) = 1;
for i = 2:10
    Data = dataString(i);
    eval(['trainImage(1+trainDataCount*(i-1):trainDataCount*i, 1:end) = double(', Data{1}, '(1:trainDataCount, 1:end));']);
    VT(1+trainDataCount*(i-1):trainDataCount*i) = i;
end
testDataCount = 800;
testImage = test0(1:testDataCount, 1:end);
VTt(1:testDataCount) = 1;
for i = 12:20
    Data = dataString(i);
    eval(['testImage(1+testDataCount*(i-11):testDataCount*(i-10), 1:end) = double(', Data{1}, '(1:testDataCount, 1:end));']);
    VTt(1+testDataCount*(i-11):testDataCount*(i-10)) = i-10;
end

% Label, The highest value is the result
T = zeros(10, trainDataCount*10);
for i = 1:trainDataCount*10
    T(VT(i), i) = 1;
end
Tt = zeros(10, testDataCount*10);
for i = 1:testDataCount*10
    Tt(VTt(i), i) = 1;
end

% Transfer to double
trainImage = im2double(trainImage);
testImage = im2double(testImage);

% PCA
[coef, ~, latent] = pca(trainImage);
lat = cumsum(latent)./sum(latent); 
a = find(lat > 0.95);
mm = a(1);
P = trainImage * coef(:, 1:mm);
Pt = testImage * coef(:, 1:mm);
P = P';
Pt = Pt';
% P = trainImage';
% Pt = testImage';

net = newff(P, T, [160 10], {'logsig','logsig'}, 'traingdx');
net.trainParam.epochs = 3000;
net.trainParam.goal = 1e-2;
net.trainParam.lr = 1;
net.trainParam.mc = 0.1;
net.divideFcn = ''; 
disp('training...');
net = train(net, P, T);

sim1 = sim(net, P);
[~ , Y1] = max(sim1);
ratio1 = mean(Y1 == VT);
disp('train ratio¡G');  disp(ratio1);
trainError = zeros(10, 10);
for i = 1:trainDataCount*10
    trainError(VT(i), Y1(i)) = trainError(VT(i), Y1(i)) + 1;
end
trainError = trainError / trainDataCount

sim2 = sim(net, Pt);
[~ , Y2] = max(sim2);
ratio2 = mean(Y2 == VTt);
disp('test ratio¡G');  disp(ratio2);
testError = zeros(10, 10);
for i = 1:testDataCount*10
    testError(VTt(i), Y2(i)) = testError(VTt(i), Y2(i)) + 1;
end
testError = testError / testDataCount