clc;
close all;
%import 'net1(rmsprop).xlsx'
train1=readtable('net1.xlsx');
train2=readtable('net2.xlsx');
train3=readtable('net3.xlsx');
plot(train1.Accuracy)
hold on
plot(train2.Accuracy,'r')
hold on
plot(train3.Accuracy,'g')
hold off
grid on
% title("Googlenet training");
title("Training Accuracies")
xlabel("Iterations")
ylabel("Accuracy (%)")
xlim([0 180])
ylim([0 101])
legend("Googlenet","Resnet-50","Efficientnet-b0")
% legend('sgdm','adam','rmsprop');
figure(2)
plot(train1.loss)
hold on
plot(train2.Loss,'r')
hold on
plot(train3.Loss,'g')
hold off
grid on
title("Training Loss")
% title("Googlenet loss");
xlabel("Iterations")
ylabel("Loss")
% legend('sgdm','adam','rmsprop');
legend("Googlenet","Resnet-50","Efficientnet-b0")