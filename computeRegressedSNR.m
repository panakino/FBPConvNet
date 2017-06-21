function [SNR, rec, c] = computeRegressedSNR(rec,oracle)

sumP    =        sum(oracle(:))           ;
sumI    =        sum(rec(:))              ;
sumIP   =        sum( oracle(:) .* rec(:) )  ;
sumI2   =        sum(rec(:).^2)           ;
A       =        [sumI2, sumI; sumI, numel(oracle)];
b       =        [sumIP;sumP]             ;
c       =        (A)\b                    ;
rec     =        c(1)*rec+c(2)            ;
err     =        sum((oracle(:)-rec(:)).^2)      ;
SNR     =        10*log10(sum(oracle(:).^2)/err) ;