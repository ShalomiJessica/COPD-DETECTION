function x = IHNN(y, frmSz, shiftAmount)
    [fftSz, ylen] = size(y);
    xlen = (ylen-1)*shiftAmount+frmSz;
    x = zeros(xlen, 1);
    
    y2 = conj(flip(y(2:fftSz-1,:)));
    y = [y; y2];
   
    sp = 1; ep = frmSz;
    for a=1:ylen
        frame = ifft(y(:,a))/(frmSz/shiftAmount/2);
        x(sp:ep) = x(sp:ep) + frame(1:frmSz);
        sp = sp + shiftAmount;
        ep = ep + shiftAmount;
    end
    x = x(1+frmSz:xlen-frmSz);
end
