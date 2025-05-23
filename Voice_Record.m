Fs=8000;
a= audiorecorder(Fs,8,1);  %Fs= 8000 Hz
disp('Start Recording...');
pause(2);
recordblocking(a,5);
disp('Stop Recording...');
x=getaudiodata(a);
filename = 'audio\user\voice.wav';  % Saving the file as MyVoice.wav
audiowrite(filename, x, Fs)
sound(x);
figure('Name','Recorded Data');subplot(2,1,1);
plot(x);
title(' Time Domain ');
xlabel('Time') % x-axis label
ylabel('Amplitude') % y-axis label
c=fft(x);
subplot(2,1,2);
plot(abs(c));
title(' Frequency Domain ');
xlabel('Frequency') % x-axis label
ylabel('Absolute Amplitude') % y-axis label
