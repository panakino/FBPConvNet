function net = cnn_unet_init(varargin)
% CNN_IMAGENET_INIT  Initialize a standard CNN for ImageNet
% Unet multi-level decomposition
opts.scale = 1 ;
opts.initBias = .05 ;
opts.weightDecay = .0001 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204*10 ; % 1GB
opts.patchSize = 50;
opts.waveLevel = 6;
opts.channel_out=1;
opts.channel_in=1;
opts = vl_argparse(opts, varargin) ;
ch_length=opts.channel_in;
net=[];
net.meta.normalization.imageSize = [opts.patchSize,opts.patchSize,ch_length] ;
 
net = unet(net, opts) ;
bs = 256 ;
 
 
% final touches
switch lower(opts.weightInitMethod)
    case {'xavier', 'xavierimproved'}
        net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
net.layers{end+1} = struct('type', 'euclideanloss', 'name', 'loss') ;
net.meta.inputSize = net.meta.normalization.imageSize ;
net.meta.augmentation.rgbVariance = zeros(0,3) ;
net.meta.augmentation.transformation = 'stretch' ;

if ~opts.batchNormalization
    lr = 1*logspace(-2, -3, 20) ;
else
    lr = logspace(-2, -3, 20) ;
end

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 1e-8 ;
 
% Fill in default values
net = vl_simplenn_tidy(net) ;
 
% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
            {'prediction','label'}, 'top1err') ;
        net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
            'opts', {'topK',5}), ...
            {'prediction','label'}, 'top5err') ;
    otherwise
        assert(false) ;
end
 
function net = add_conv(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
    name = 'fc' ;
else
    name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
    'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
    'stride', stride, ...
    'pad', pad, ...
    'learningRate', [1 2], ...
    'weightDecay', [opts.weightDecay 0], ...
    'opts', {convOpts}) ;
 
 
% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad,batchOn,ReluOn)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
    name = 'fc' ;
else
    name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
    'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
    'stride', stride, ...
    'pad', pad, ...
    'dilate', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [opts.weightDecay 0], ...
    'opts', {convOpts}) ;
if (opts.batchNormalization)&&batchOn
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
        'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
        'learningRate', [2 1 0.05], ...
        'weightDecay', [0 0]) ;
end
if ReluOn
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end 

% --------------------------------------------------------------------
function net = add_block_convt(net, opts, id, h, w, in, out, upsample, crop,batchOn,ReluOn)
% --------------------------------------------------------------------
% crop=[2 3 2 3];
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
    name = 'fc' ;
else
    name = 'convt' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'convt', 'name', sprintf('%s%s', name, id), ...
    'weights', {{init_weight(opts, h, w, out, in, 'single'), zeros(out, 1, 'single')}}, ...
    'upsample', upsample, ...
    'crop', crop, ...
    'learningRate', [1 2], ...
    'weightDecay', [opts.weightDecay 0], ...
    'opts', {convOpts}) ;
if (opts.batchNormalization)&&batchOn
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
        'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
        'learningRate', [2 1 0.05], ...
        'weightDecay', [0 0]) ;
end
if ReluOn
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end 

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.
 
switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
 
% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
    net.layers{end+1} = struct('type', 'normalize', ...
        'name', sprintf('norm%s', id), ...
        'param', [5 1 0.0001/5 0.75]) ;
end
 
% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'dropout', ...
    'name', sprintf('dropout%s', id), ...
    'rate', 0.5) ;
 
function net = add_reg_catch(net, id,regNum,reluOn)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'reg_catch', ...
    'name', sprintf('reg%s', id),'regNum',regNum) ;
if reluOn
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end
 
function net = add_reg_toss(net, id, regNum)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'reg_toss', ...
    'name', sprintf('reg%s', id),...
    'regNum', regNum) ;

function net = add_reg_concat(net, id, regSet)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'reg_concat', ...
    'name', sprintf('reg%s', id),...
    'regSet', regSet) ;


% --------------------------------------------------------------------
function net = add_pool(net, opts,id, stride, pad)
% --------------------------------------------------------------------
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'pool', 'name', sprintf('pool%s', id), ...
    'pool',stride,...
    'stride', stride, ...
    'pad', pad, ...
    'method','max',...
    'opts', {convOpts}) ;

% --------------------------------------------------------------------
function net = add_upconv(net, id, stride, pad)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'upconv', 'name', sprintf('upconv%s', id), ...
    'stride', stride, ...
    'pad', pad) ;


% --------------------------------------------------------------------
function net = unet(net, opts)
% --------------------------------------------------------------------
 
net.layers = {} ;
ch_length = opts.channel_in;
ch_length_out = opts.channel_out;
KerenlSize = 3;
zeroPad = floor(KerenlSize/2);

net = add_block(net, opts, '0', KerenlSize, KerenlSize, ch_length, 64, 1, zeroPad,1,1) ;
net = add_block(net, opts, '0', KerenlSize, KerenlSize, 64, 64, 1, zeroPad,1,1) ;
net = add_block(net, opts, '0', KerenlSize, KerenlSize, 64, 64, 1, zeroPad,1,1) ;
net = add_reg_toss(net, '0',1);
net = add_pool(net, opts,'0', 2, 0);


net = add_block(net, opts, '1_1', KerenlSize, KerenlSize, 64, 128, 1, zeroPad,1,1) ;
net = add_block(net, opts, '1_2', KerenlSize, KerenlSize, 128, 128, 1, zeroPad,1,1) ;
net = add_reg_toss(net, '1',2);
net = add_pool(net, opts,'1', 2, 0);


net = add_block(net, opts, '2_1', KerenlSize, KerenlSize, 128, 256, 1, zeroPad,1,1) ;
net = add_block(net, opts, '2_2', KerenlSize, KerenlSize, 256, 256, 1, zeroPad,1,1) ;
net = add_reg_toss(net,'2',3);
net = add_pool(net, opts,'2', 2, 0);


net = add_block(net, opts, '3_1', KerenlSize, KerenlSize, 256, 512, 1, zeroPad,1,1) ;
net = add_block(net, opts, '3_2', KerenlSize, KerenlSize, 512, 512, 1, zeroPad,1,1) ;
net = add_reg_toss(net,'3',4);
net = add_pool(net, opts,'3',2,0);


net = add_block(net, opts, '4_1', KerenlSize, KerenlSize, 512, 1024, 1, zeroPad,1,1) ;
net = add_block(net, opts, '4_2', KerenlSize, KerenlSize, 1024, 1024, 1, zeroPad,1,1) ;
net = add_block_convt(net, opts, '4_3', KerenlSize, KerenlSize, 1024, 512, 2, [0 1 0 1],1,1) ;

net = add_reg_concat(net, '5_0',4);
net = add_block(net, opts, '5_1', KerenlSize, KerenlSize, 1024, 512, 1, zeroPad,1,1) ;
net = add_block(net, opts, '5_2', KerenlSize, KerenlSize, 512, 512, 1, zeroPad,1,1) ;
net = add_block_convt(net, opts, '5_3', KerenlSize, KerenlSize, 512, 256, 2,  [0 1 0 1],1,1) ;


net = add_reg_concat(net, '6_0',3);
net = add_block(net, opts, '6_1', KerenlSize, KerenlSize, 512, 256, 1, zeroPad,1,1) ;
net = add_block(net, opts, '6_2', KerenlSize, KerenlSize, 256, 256, 1, zeroPad,1,1) ;
net = add_block_convt(net, opts, '6_3', KerenlSize, KerenlSize, 256, 128, 2,  [0 1 0 1],1,1) ;


net = add_reg_concat(net, '7_0',2);
net = add_block(net, opts, '7_1', KerenlSize, KerenlSize, 256, 128, 1, zeroPad,1,1) ;
net = add_block(net, opts, '7_2', KerenlSize, KerenlSize, 128, 128, 1, zeroPad,1,1) ;
net = add_block_convt(net, opts, '7_3', KerenlSize, KerenlSize, 128, 64, 2,  [0 1 0 1],1,1) ;

net = add_reg_concat(net, '8_0',1);
net = add_block(net, opts, '8_1', KerenlSize, KerenlSize, 128, 64, 1, zeroPad,1,1) ;
net = add_block(net, opts, '8_2', KerenlSize, KerenlSize, 64, 64, 1, zeroPad,1,1) ;
net = add_block(net, opts, '8_3', 1, 1, 64, ch_length_out, 1, 0,0,0) ;


info = vl_simplenn_display(net);
net.meta.regNum = 4;
net.meta.regSize = [info.dataSize(1,end),info.dataSize(2,end),info.dataSize(3,end)]; 
 

