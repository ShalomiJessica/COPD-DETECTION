function [EstMag] = HNN_ANALYSIS(Mag, initialNoiseLen, alpha)
    Pow = Mag.^2;
    [nfft, nfrm] = size(Pow);

    noisePow = zeros(nfft, 1);
    for b=1:nfft
        noisePow(b) = mean(Pow(b,2:1+initialNoiseLen));
    end

    if ~exist('alpha', 'var')
        alpha = 0.98;
    end
    aposterioriSNR = zeros(nfft, nfrm);
    for b=1:nfft
       aposterioriSNR(b,:) = Pow(b,:)/noisePow(b);
    end
    aprioriSNR = alpha + (1-alpha)*max(aposterioriSNR-1,0);

    EstMag = zeros(nfft, nfrm);
    for a=2:nfrm-1
        vk = aprioriSNR(:,a)./(1+aprioriSNR(:,a)).*aposterioriSNR(:,a);
        I0 = besseli(0, vk./2);
        I1 = besseli(1, vk./2);
        gain = (1+vk).*I0 + vk.*I1;
        gain = gain.*exp(-vk/2).*sqrt(vk)./aposterioriSNR(:,a);
        gain = gain*sqrt(pi)/2;
        gain(isnan(gain)) = 1;
        gain(isinf(gain)) = 1;
        EstMag(:,a) = Mag(:,a).*gain;
        aprioriSNR(:,a+1) = alpha*EstMag(:,a).^2./noisePow(:) + (1-alpha)*max(aposterioriSNR(:,a+1)-1,0);
    end
end
