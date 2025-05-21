% This STFT function use Hanning window
% frmSz is the size of acoustic frame
% fftSample determines sampling rate for the continous DTFT, for example,
% fftSample = 1 with frmSz = 256, the result will be 128-points DFT

function y = HNN(x, frmSz, shiftAmount, fftSample)
    fftSz = fftSample * frmSz;
    hwin = zeros(frmSz, 1);
    for a=1:frmSz
        hwin(a, 1) = 0.5 - 0.5*cos(2*pi*(a-1+0.5)/frmSz);
    end
    x = padarray(x, frmSz);
    xlen = length(x);
    
    sp = 0; a = 0;
    while sp + frmSz <= xlen
        sp = sp + shiftAmount;
        a = a + 1;
    end
    y = zeros(fftSz/2+1, a);

    sp = 1; a = 1; ep = frmSz;
    while ep <= xlen
        z = fft(x(sp:ep).*hwin, fftSz);
        y(:,a) = z(1:fftSz/2+1);
        sp = sp + shiftAmount;
        ep = ep + shiftAmount;
        a = a + 1;
    end
end
