function heatmap = simplenn_graphical_deconvolution(net, im, im_, varargin)
%SIMPLENN_GRAPHICAL_DECONVOLUTION. Deconvoluting activations back to the input layer.
%   HEATMAP = SIMPLENN_GRAPHICAL_DECONVOLUTION(NET, IM, IM_)
%   deconvolutes activations to generate a heatmap of activations,
%   and displays it to the user.
%   NET. The CNN to visualize.
%   IM. The original input image. Needed for user output only.
%   IM_. The adjusted input image used as input to the CNN. If IM_ is a
%        gpuArray computations are run on the GPU.
%
%   SIMPLENN_GRAPHICAL_DECONVOLUTION(...,'OPT',VALUE,...) takes the following options:
%
%   'ReLUPass':: 'Guided Backpropagation'
%       Sets the method used to deconvolute activations through the ReLU layers.
%       The available methods are 'Backpropagation', 'Deconvnet', and 'Guided
%       Backpropagation'. The default method is 'Guided Backpropagation'
%       since this method usually gives the best results.
%
%   'ConvolutionPass':: 'Standard'
%       Sets the method used to deconvolute activations through the convolution
%       layers. The available methods are 'Relevance Propagation', and
%       'Standard'. The default method is 'Standard', since this
%       method usually gives the best results.
%
%   'MeasureLayer':: Last layer
%       An Int32 specifying the layer from which activations should be
%       deconvoluted back to the input layer. By default the last layer of
%       the network is used.
%
%   'MeasureFilter':: Strongest activated filter
%       An Int32 specifying the filter for which activations should be
%       deconvoluted back to the input layer. By default the strongest
%       activated filter is used.
%
%   ADVANCED USAGE
%
%   Computations are run on the GPU if IM_ is a gpuArray. Otherwise they
%   are run on the CPU.
%

% Copyright (C) 2016 Felix Grün.
% All rights reserved.
%
% This file is part of the FeatureVis library and is made available under
% the terms of the BSD license (see the LICENCE file).


    gpuMode = isa(im_, 'gpuArray') ;

    % move everything to the GPU
    if gpuMode
        net = vl_simplenn_move(net, 'gpu');
    else
        net = vl_simplenn_move(net, 'cpu');
    end

    % forward pass with dropout disabled
    res = vl_simplenn(net, im_, [], [], 'Mode', 'test', 'conserveMemory', false) ;

    % needed to display the classification result of the network
    scores = squeeze(gather(res(end).x)) ;
    [classScore, class] = max(scores) ;

    heatmap =  simplenn_deconvolution(net, im_, varargin{:});

    % --- display results ---
    display(net, im, heatmap, class, classScore);
end

function display(net, im, heat, class, classScore)

    % --- output results ---

    % filter for the positive activations
    heat = heat .* (heat > double(0));

    % normalize per pixel over all color channles
    for w = 1:size(heat,2)
        for h = 1:size(heat,1)
            heat(h,w,:) = norm(squeeze(heat(h,w,:)));
        end
    end
    heat = heat / max(heat(:));

    % --- "stretch" heatmap to the size of the original image ---

    % calculate the resize factors for width and height
    fac_h = cast(size(im, 1), 'double') / cast(size(heat, 1), 'double');
    fac_w = cast(size(im, 2), 'double') / cast(size(heat, 2), 'double');

    % pre-allocate the resized heatmap
    im2 = zeros(size(im,1), size(im,2), size(im,3), 'double');

    % resize the heatmap
    for h = 1:size(heat,1)
        for w = 1:size(heat,2)
            im2(round((h-1)*fac_h)+1:round(h*fac_h), round((w-1)*fac_w)+1:round(w*fac_w), 1) = heat(h,w);
        end
    end

    % set heatmap values for all color channels
    if (size(im,3) > 1)
        for h = 2:size(im,3)
            im2(:,:,h) = im2(:,:,1);
        end
    end

    % --- output results ---

    im4 = ind2rgb(double(round(squeeze(im2(:,:,1)) * 255)), jet(255));
    im4(:) = cast(round(cast(im4(:), 'double') * 255), 'uint8');

    % create overlay
    im3 = zeros(size(im,1), size(im,2), size(im,3), 'uint8');
    im3(:) = cast(round((cast(im4, 'double') * 0.6) + (cast(im, 'double') * 0.4)), 'uint8');

    % display image, heatmap, and overlay
    titleText = '';

    if isfield(net, 'classes')
        titleText = net.classes.description{class};
    elseif isfield(net, 'meta')
        titleText = net.meta.classes.description{class};
    end

    rows = size(im, 1);
    cols = size(im, 2);

    if size(im, 1) < 200 && size(im, 2) < 200
        rows = size(im, 1) * 4;
        cols = size(im, 2) * 4;
    elseif size(im, 1) < 300 && size(im, 2) < 300
        rows = size(im, 1) * 3;
        cols = size(im, 2) * 3;
    elseif size(im, 1) < 400 && size(im, 2) < 400
        rows = size(im, 1) * 2;
        cols = size(im, 2) * 2;
    elseif size(im, 1) < 500 && size(im, 2) < 600
        rows = round(size(im, 1) * 1.5);
        cols = round(size(im, 2) * 1.5);
    end

    vec = [rows, cols];

    fig1 = figure('Name', 'Original Image'); clf; imagesc(im); title(sprintf('%s: %.3f', titleText, classScore)); truesize(fig1, vec);
    fig2 = figure('Name', 'Black and White Heat Map'); clf; imagesc(im2); truesize(fig2, vec);
    fig3 = figure('Name', 'Overlay'); clf; imagesc(im3); truesize(fig3, vec);
    fig4 = figure('Name', 'Color Heat Map'); clf; imagesc(im4); truesize(fig4, vec);
end
