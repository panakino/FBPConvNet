function [net, info] = cnn_fbpconvnet(varargin)


%sparsifying filter
opts.expDir='./training_result';
std_noise = 30;
opts.weightInitMethod = 'gaussian' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.whitenData = false ;
opts.lite = false ;
opts.contrastNormalization = false ;
opts.train = struct() ;
opts.useGpu = true;
opts.gpus = 1 ;
opts.patchSize = 50;
opts.batchSize = 10 ;
opts.numEpochs = 1000 ;
opts.std_noise = std_noise ;
opts.lambda = 1e-4;
opts.imdb = [];
opts.waveLevel = 6;
opts.waveName = 'vk';
opts.gradMax = 1e-2;
opts.channel_out=1;
opts.channel_in=1;
opts.weight='none';
opts.plotStatistics=0;

opts.momentum = 0.9 ;
opts = vl_argparse(opts, varargin) ;
% if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

net = cnn_unet_init( 'batchNormalization', opts.batchNormalization, ...
    'weightInitMethod', opts.weightInitMethod,'patchSize',opts.patchSize,'waveLevel',opts.waveLevel,...
    'channel_in',opts.channel_in,'channel_out',opts.channel_out);



% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
if isempty(opts.imdb) 
    imdb = load(opts.imdbPath) ;
else 
    imdb = opts.imdb;
end 
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainFn = @cnn_train_fbpconvnet;
    case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 2),...
     'batchSize', opts.batchSize,...
     'numEpochs',opts.numEpochs,...
     'batchSize',opts.batchSize,...
     'gpus',opts.gpus,...
     'lambda',opts.lambda,...
     'waveLevel',opts.waveLevel,...
     'waveName',opts.waveName,...
     'weight',opts.weight,...
     'gradMax', opts.gradMax,'momentum',opts.momentum) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
fn = @(x,y) getSimpleNNBatch(x,y,opts.patchSize) ;


% -------------------------------------------------------------------------
function [images, labels, lowRes] = getSimpleNNBatch(imdb, batch, patchSize)
% -------------------------------------------------------------------------
Ny = size(imdb.images.noisy,1);
Nx = size(imdb.images.noisy,2);
pos_x = round(rand(1)*(Nx-patchSize));
pos_y = round(rand(1)*(Ny-patchSize));
images = single(imdb.images.noisy(pos_y+(1:patchSize),pos_x+(1:patchSize),:,batch)) ;
labels = single(imdb.images.orig(pos_y+(1:patchSize),pos_x+(1:patchSize),:,batch)) ;
if rand > 0.5
    labels=fliplr(labels);
    images=fliplr(images);
end
if rand > 0.5
    labels=flipud(labels);
    images=flipud(images);
end
lowRes = images(:,:,1,:);
labels(:,:,1,:) = labels(:,:,1,:) - lowRes;

