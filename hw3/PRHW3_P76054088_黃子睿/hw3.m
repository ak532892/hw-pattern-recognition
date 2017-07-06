clc
clearvars

% Train data
M = 300; % size
d = 2; % dim
N1 = normrnd(-1, 1, [d M/2]);
N2 = normrnd(1, 1, [d M/2]);
N = [N1 N2];
T(1:M/2) = 0;
T(M/2+1:M) = 1;

% Test data
Mt = 100; % size
N3 = normrnd(-1, 1, [d Mt/2]);
N4 = normrnd(1, 1, [d Mt/2]);
Nt = [N3 N4];
Tt(1:Mt/2) = 0;
Tt(Mt/2+1:Mt) = 1;

n = 5;
net1 = feedforwardnet(n-1);
net2 = feedforwardnet(n);
net3 = feedforwardnet(n+1);
net1.layers{1}.transferFcn = 'logsig';
net2.layers{1}.transferFcn = 'logsig';
net3.layers{1}.transferFcn = 'logsig';

net1 = train(net1, N, T); 
net2 = train(net2, N, T);
net3 = train(net3, N, T);

%net1
% train data test
yt = net1(N);

plot(yt, 'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count = 0;
for i = 1:M
    if yt(i) < 0.5
		yt(i) = 0;
    else
		yt(i) = 1;
    end
	
    if yt(i) == T(i)
		count = count + 1;
    end
end

accuracy = count / M * 100;
title(['train data accuracy = ', num2str(accuracy), '%']);

% test data test
yt = net1(Nt);

figure(2);

plot(yt, 'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count = 0;
for i = 1:Mt
    if yt(i) < 0.5
		yt(i) = 0;
    else
		yt(i) = 1;
    end
	
    if yt(i) == Tt(i)
		count = count + 1;
    end
end

accuracy = count / Mt * 100;
title(['test data accuracy = ', num2str(accuracy), '%']);

%net2
% train data test
yt = net2(N);

figure(3);
plot(yt, 'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count = 0;
for i = 1:M
    if yt(i) < 0.5
		yt(i) = 0;
    else
		yt(i) = 1;
    end
	
    if yt(i) == T(i)
		count = count + 1;
    end
end

accuracy = count / M * 100;
title(['train data accuracy = ', num2str(accuracy), '%']);

% test data test
yt = net2(Nt);

figure(4);
plot(yt, 'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count = 0;
for i = 1:Mt
    if yt(i) < 0.5
		yt(i) = 0;
    else
		yt(i) = 1;
    end
	
    if yt(i) == Tt(i)
		count = count + 1;
    end
end

accuracy = count / Mt * 100;
title(['test data accuracy = ', num2str(accuracy), '%']);

%net3
% train data test
yt = net3(N);

figure(5);
plot(yt, 'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count = 0;
for i = 1:M
    if yt(i) < 0.5
		yt(i) = 0;
    else
		yt(i) = 1;
    end
	
    if yt(i) == T(i)
		count = count + 1;
    end
end

accuracy = count / M * 100;
title(['train data accuracy = ', num2str(accuracy), '%']);

% test data test
yt = net3(Nt);

figure(6);
plot(yt, 'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count = 0;
for i = 1:Mt
    if yt(i) < 0.5
		yt(i) = 0;
    else
		yt(i) = 1;
    end
	
    if yt(i) == Tt(i)
		count = count + 1;
    end
end

accuracy = count / Mt * 100;
title(['test data accuracy = ', num2str(accuracy), '%']);