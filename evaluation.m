% evaluation - FBPConvNet
% modified from MatconvNet (ver.23)
% 22 June 2017
% contact : Kyong Jin (kyonghwan.jin@gmail.com)

clear
restoredefaultpath
reset(gpuDevice(1))
run ./matconvnet-1.0-beta23/matlab/vl_setupnn

load preproc_x20_ellipse_fullfbp.mat
load('./pretrain/net-epoch-151.mat')

cmode='gpu'; % 'cpu'
if strcmp(cmode,'gpu')
    net = vl_simplenn_move(net, 'gpu') ;
else
    net = vl_simplenn_move(net, 'cpu') ;
end

avg_psnr_m=zeros(25,1);
avg_psnr_rec=zeros(25,1);
for iter=476:500
    gt=lab_n(:,:,1,iter);
    m=lab_d(:,:,1,iter);
    if strcmp(cmode,'gpu')
        res=vl_simplenn_fbpconvnet_eval(net,gpuArray((single(m))));
        rec=gather(res(end-1).x)+m;
    else
        res=vl_simplenn_fbpconvnet_eval(net,((single(m))));
        rec=(res(end-1).x)+m;
    end
    
    snr_m=computeRegressedSNR(m,gt);
    snr_rec=computeRegressedSNR(rec,gt);
    figure(1), 
    subplot(131), imagesc(m),axis equal tight, title({'fbp';num2str(snr_m)})
    subplot(132), imagesc(rec),axis equal tight, title({'fbpconvnet';num2str(snr_rec)})
    subplot(133), imagesc(gt),axis equal tight, title(['gt ' num2str(iter)])
    pause(0.1)
    
    avg_psnr_m(iter-475)=snr_m;
    avg_psnr_rec(iter-475)=snr_rec;
end

display(['avg SNR (FBP) : ' num2str(mean(avg_psnr_m))])
display(['avg SNR (FBPconvNet) : ' num2str(mean(avg_psnr_rec))])
