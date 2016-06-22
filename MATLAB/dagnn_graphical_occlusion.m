function heatmap = dagnn_graphical_occlusion(net, im, im_, varargin)
%DAGNN_GRAPHICAL_OCCLUSION. Generating a heatmap of feature importance.
%   HEATMAP = DAGNN_GRAPHICAL_OCCLUSION(NET, IM, IM_, {SIZE, STRIDE})
%   generates a heatmap showing the importance of different areas of the input image,
%   and displays it to the user.
%   NET. The CNN to visualize.
%   IM. The original input image. Needed for user output only.
%   IM_. The adjusted input image used as input to the CNN. If IM_ is a
%        gpuArray computations are run on the GPU.
%   SIZE. The size (side length) of the occluded area.
%   STRIDE. The stride width with which the occluded area will be moved
%   accross the occluded filter.
%
%   DAGNN_GRAPHICAL_OCCLUSION(...,'OPT',VALUE,..., {SIZE, STRIDE}) takes the following options:
%
%   'InputName':: Empty
%      Sets the input variable. Only required for dag networks with more than
%      one input variable.
%
%   'OutputName':: Empty
%      Sets the output variable. Only required for dag networks with more than
%      one output variable.
%
%   'MeasureLayer':: Last layer
%       An Int32 specifying the layer at which the changes in activations
%       should be measured. By default the last layer or, if the last layer
%       is a softmax layer, the second to last layer of the network is used.
%
%   'MeasureFilter':: Strongest activated filter
%       An Int32 specifying the filter at which changes in activations
%       should be measured. By default the strongest activated filter is
%       used.
%
%   'BoxColor'::  Random pixel values
%       A 1x3 single Array specifying the color to be used for the occlusion box.
%       The three values correspond to the three color channels.
%
%   ADVANCED USAGE
%
%   Computations are run on the GPU if IM_ is a gpuArray. Otherwise they
%   are run on the CPU.
%   Size and stride of the occluded area are always specified last.
%   The two values for size and stride may be replaced by four values where
%   the first two values specify the size and stride along the width of the
%   input (horizontally) and the last two values specify the size and
%   stride along the height of the input (vertically). In fact any number of
%   cell arrays of two or four values may be used to run more than one test
%   and average the results, e.g. two cell arrays might specify one test with
%   a larger occluded area and stride rate first and a second test with a
%   smaller occluded area and stride rate last.
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

    % a boxColor value of 0 means random values
    boxColor = 0;

    % parse optional parameters
    for i = 1:2:length(varargin)

        if ~ischar(varargin{i})
            break;
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
        case 'boxcolor'
            if ~isnumeric(varargin{i+1}) error('The values for boxColor must be of type single'); end
            if ~length(varargin{i+1}(:)) == 3 && ~size(varargin{i+1},1) == 1 && ~size(varargin{i+1},2) == 3
                error('The value for boxColor must be an 1x3 single Array');
            end
            boxColor = single(varargin{i+1});
            parameters{end+1} = varargin{i};
            parameters{end+1} = varargin{i+1};
        otherwise
            parameters{end+1} = varargin{i};
            parameters{end+1} = varargin{i+1};
        end
    end

    % save to retrieve size and stride values later
    index = i;

    % add size and stride values to the parameters
    s = varargin(i:end);
    parameters((end+1):(end+length(s))) = s(:);

    % check input
    for i = index:length(varargin)
        if isempty(varargin{i}) || (length(varargin{i}) ~= 2 && length(varargin{i}) ~= 4)
            error('Either provide one size and one stride value or provide four values (size_w, stride_w, size_h, stride_h) for each run.');
        end
    end

    % If output variable has not been set, set if to the first output of the
    % layer where non of the outputs is an input to another layer
    if outputVariable < 0
        for i = length(net.layers):-1:1
            out = net.layers(i).outputIndexes;
            onlyOutputs = true;
            for j = 1:length(out)
                if net.vars(out(j)).fanout ~= 0
                    onlyOutputs = false;
                    break;
                end
            end
            if onlyOutputs
                outputVariable = out(1);
                outputName =  net.vars(outputVariable).name;
                break;
            end
        end
    end

    % If input variable has not been set, set it to the first input of the layer
    % where non of the inputs is an output of another layer
    if inputVariable < 0
        for i = 1:length(net.layers)
            in = net.layers(i).inputIndexes;
            onlyInputs = true;
            for j = 1:length(in)
                if net.vars(in(j)).fanin ~= 0
                    onlyInputs = false;
                    break;
                end
            end
            if onlyInputs
                inputVariable =  in(1);
                inputName = net.vars(inputVariable).name;
                break;
            end
        end
    end

    gpuMode = isa(im_, 'gpuArray') ;

    % move everything to the GPU
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
    scores = net.vars(outputVariable).value ;
    scores = squeeze(gather(scores)) ;
    [classScore, classIndex] = max(scores) ;

    % display example occlusion
    % this can help if the occlusion box color is set manually
    if length(varargin{index}) == 2
        size_w = varargin{index}{1};
        size_h = varargin{index}{1};
    else
        size_w = varargin{index}{1};
        size_h = varargin{index}{3};
    end

    im2 = im_ ;
    randomColor = length(boxColor) ~= 3;

    % overlay the occlusion box over the copied image
    if randomColor
        im2(1:size_h, 1:size_w, :) = rand(size_h, size_w, size(im2,3), 'single') * 256 - 128;
    else
        im2(1:size_h, 1:size_w, 1) = boxColor(1) ;
        im2(1:size_h, 1:size_w, 2) = boxColor(2) ;
        im2(1:size_h, 1:size_w, 3) = boxColor(3) ;
    end

    im2 = (im2 - min(im2(:))) / (max(im2(:)) - min(im2(:)));
    fig = figure('Name', 'Top left area occluded'); clf; imagesc(im2); truesize(fig);

    heatmap = dagnn_occlusion(net, im_, inputName, outputName, parameters{:});

    % --- display results ---
    display(net, im, heatmap, classIndex, classScore);
end

function display(net, im, heat, class, classScore)

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
    im3(:) = cast(round((cast(im4, 'double') / 2) + (cast(im, 'double') / 2)), 'uint8');

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
