function heatmap = dagnn_graphical_deconvolution(net, im, im_, varargin)
%DAGNN_GRAPHICAL_DECONVOLUTION. Deconvoluting activations back to the input layer.
%   HEATMAP = DAGNN_GRAPHICAL_DECONVOLUTION(NET, IM, IM_)
%   deconvolutes activations to generate a heatmap of activations,
%   and displays it to the user.
%   NET. The CNN of type DDagNN to visualize.
%   IM. The original input image. Needed for user output only.
%   IM_. The adjusted input image used as input to the CNN. If IM_ is a
%        gpuArray computations are run on the GPU.
%
%   DAGNN_GRAPHICAL_DECONVOLUTION(...,'OPT',VALUE,...) takes the following options:
%
%   'InputName':: Empty
%      Sets the input variable. Only required for dag networks with more than
%      one input variable.
%
%   'OutputName':: Empty
%      Sets the output variable. Only required for dag networks with more than
%      one output variable.
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


    % --- process input ---

    % set standard values for optional parameters
    inputName = '';
    inputVariable = -1;
    outputName = '';
    outputVariable = -1;
    parameters = cell(0);

    % parse optional parameters
    for i = 1:2:length(varargin)

        if ~ischar(varargin{i})
            error('The %d. parameter name is not of type char', fix(i/2));
        end

        switch lower(varargin{i})
        case 'inputname'
            if ~ischar(varargin{i+1}) error('The value for inputName must be a char'); end
            inputName = varargin{i+1};
            inputVariable = net.getVarIndex(inputName);
            if isnan(inputVariable)
                error('There is no variable with name %s', inputName);
            end
        case 'outputname'
            if ~ischar(varargin{i+1}) error('The value for outputName must be a char'); end
            outputName = varargin{i+1};
            outputVariable = net.getVarIndex(outputName);
            if isnan(outputVariable)
                error('There is no variable with name %s', outputName);
            end
        otherwise
            parameters{end+1} = varargin{i};
            parameters{end+1} = varargin{i+1};
        end
    end

    % If output variable has not been set, set if to the first output variable
    % of the output layer
    if outputVariable < 0
        for i = length(net.layers):-1:1

            onlyOutputs = true;

            for out = net.layers(i).outputIndexes
                if net.vars(out).fanout
                    onlyOutputs = false;
                    break; % inner loop
                end
            end

            if onlyOutputs
                outputVariable = net.layers(i).outputIndexes(1);
                outputName = net.vars(outputVariable).name;
                break; % outter loop
            end
        end
    end

    % If input variable has not been set, set it to the first input variable
    % of the input layer
    if inputVariable < 0
        for i = 1:length(net.layers)

            onlyInputs = true;

            for in = net.layers(i).inputIndexes
                if net.vars(in).fanin
                    onlyInputs = false;
                    break; % inner loop
                end
            end

            if onlyInputs
                inputVariable = net.layers(i).inputIndexes(1);
                inputName = net.vars(inputVariable).name;
                break; % outter loop
            end
        end
    end

    heatmap = dagnn_deconvolution(net, im_, inputName, outputName, parameters{:});

    gpuMode = isa(im_, 'gpuArray') ;

    % move net to the GPU
    if gpuMode
        net.move('gpu');
    else
        net.move('cpu');
    end

    % disable dropout
    mode = net.mode;
    net.mode = 'test';

    % forward pass
    net.eval({inputName, im_});

    % reset dropout
    net.mode = mode;

    % needed to display the classification result of the network
    scores = net.vars(outputVariable).value;
    scores = squeeze(gather(scores));
    [classScore, classIndex] = max(scores);

    % --- display results ---
    display(net, im, heatmap, classIndex, classScore);
end

function display(net, im, heat, class, classScore)

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

    if (isprop(net, 'classes'))
        titleText = net.classes.description{class};
    elseif (isprop(net, 'meta'))
        if (isprop(net.meta, 'classes'))
            titleText = net.meta.classes.description{class};
        else
            disp('No description for classes found.');
        end
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
