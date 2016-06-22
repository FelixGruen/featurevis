function heatmap = dagnn_occlusion(net, im_, inputName, outputName, varargin)
%DAGNN_OCCLUSION. Generating a heatmap of feature importance.
%   HEATMAP = DAGNN_OCCLUSION(NET, IM_, INPUTNAME, OUTPUTNAME, {SIZE, STRIDE})
%   generates a heatmap showing the importance of different areas of the input image,
%   and returns the heatmap.
%   NET. The CNN to visualize.
%   IM_. The adjusted input image used as input to the CNN. If IM_ is a
%        gpuArray computations are run on the GPU.
%   INPUTNAME. The name of the input variable of the network.
%   OUTPUTNAME. The name of the output variable of the network.
%   SIZE. The size (side length) of the occluded area.
%   STRIDE. The stride width with which the occluded area will be moved
%   accross the occluded filter.
%
%   DAGNN_OCCLUSION(...,'OPT',VALUE,..., {SIZE, STRIDE}) takes the following options:
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
%   'BoxColor':: Random pixel values
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

    if ~isa(net, 'DDagNN')
        if isa(net, 'dagnn.DagNN')
            error('Please load you DagNN as DDagNN!');
        else
            error('The network is not a DagNN!');
        end
    end

    outputVariable = net.getVarIndex(outputName);
    if isnan(outputVariable)
        error('There is no variable with name %s', outputName);
    end

    inputVariable = net.getVarIndex(inputName);
    if isnan(inputVariable)
        error('There is no variable with name %s', inputName);
    end

    % set standard values for optional parameters
    measureLayer = 0;
    measureFilter = 0;

    % a boxColor value of 0 means random values
    boxColor = 0;

    % parse optional parameters
    for i = 1:2:length(varargin)

        if ~ischar(varargin{i})
            break;
        end

        switch lower(varargin{i})
            case 'measurelayer'
                if ~isnumeric(varargin{i+1}) error('The value for measureLayer must be an integer'); end
                measureLayer = cast(varargin{i+1}, 'int32');
            case 'measurefilter'
                if ~isnumeric(varargin{i+1}) error('The value for measureFilter must be an integer'); end
                measureFilter = cast(varargin{i+1}, 'int32');
            case 'boxcolor'
                if ~isnumeric(varargin{i+1}) error('The values for boxColor must be of type single'); end
                if ~length(varargin{i+1}(:)) == 3 && ~size(varargin{i+1},1) == 1 && ~size(varargin{i+1},2) == 3
                    error('The value for boxColor must be an 1x3 single Array');
                end
                boxColor = single(varargin{i+1});
            otherwise
                error('Unknown parameter "%s"', varargin{i});
        end
    end

    % the index of the size and stride values
    index = i;

    % check input
    for i = index:length(varargin)
        if isempty(varargin{i}) || (length(varargin{i}) ~= 2 && length(varargin{i}) ~= 4)
            error('Either provide one size and one stride value or provide four values (size_w, stride_w, size_h, stride_h) for each run.');
        end
    end

    if measureLayer > numel(net.layers)
        error('Measure layer (%d) cannot be greater than the number of layers in the network (%d)', measureLayer, numel(net.layers));
    elseif measureLayer < 1
        measureVariable = outputVariable;
        for i = length(net.layers):-1:1
            out = net.layers(i).outputIndexes;
            if ismember(outputVariable, out)
                in = net.layers(i).inputIndexes;
                out = net.layers(i-1).outputIndexes;
                if isa(net.layers(i).block, 'dagnn.SoftMax') && length(in) == 1 && length(out) == 1 && out == in
                    measureLayer = i-1;
                    measureVariable =  out(1);
                    break;
                else
                    measureLayer = i;
                    break;
                end
            end
        end
    end

    if measureFilter < 0
        error('Measure filter (%d) cannot be negativ', measureFilter);
    end

    measureVariable = net.getVarIndex(net.layers(measureLayer).outputs);
    if isnan(measureVariable)
        error('The measure layer (%d) has no output variable', measureLayer);
    end

    % --- preparations ---

    gpuMode = isa(im_, 'gpuArray') ;

    % move everything to the GPU
    if gpuMode
        net.move('gpu');
    else
        net.move('cpu');
    end

    % disable dropout and memory conservation
    mode = net.mode;
    net.mode = 'test';
    conserveMemory = net.conserveMemory;
    net.conserveMemory = false;

    % run the CNN
    net.eval({inputName, im_});

    % if not set by user, set the measure filter to the filter with the
    % maximum activation
    scores = net.vars(measureVariable).value;
    scores = gather(scores) ;
    if measureFilter == 0 && outputVariable == measureVariable
        [~, classIndex] = max(squeeze(scores));
        measureFilter = classIndex;
    elseif measureFilter == 0
        measureFilter = getFilter(scores);
    end

    if measureFilter > size(scores,3)
        error('Measure filter (%d) cannot be greater than the number of filters in layer %d (%d)', ...
            measureFilter, measureLayer, size(scores,3));
    end

    % Display user information
    fprintf('Measuring change in activation at layer %d (%s) filter %d.\n', ...
        measureLayer, net.layers(measureLayer).name, measureFilter);

    if isa(net.layers(measureLayer).block, 'dagnn.SoftMax')
        fprintf('Measuring the change in activation in the results of the softmax layer is discouraged!\n');
    end

    % the activation matrix of the measure filter will be needed to
    % calculate the differences in activation
    score = squeeze(scores(:,:,measureFilter));

    % reformat size and strides
    sizesAndStrides = {};

    for i = index:length(varargin)

        sizesAndStrides{end+1} =  varargin{i}{1};
        sizesAndStrides{end+1} =  varargin{i}{2};

        if length(varargin{i}) == 2
            sizesAndStrides{end+1} =  varargin{i}{1};
            sizesAndStrides{end+1} =  varargin{i}{2};
        else
            sizesAndStrides{end+1} =  varargin{i}{3};
            sizesAndStrides{end+1} =  varargin{i}{4};
        end
    end

    heatmap = occlude(net, inputName, im_, score, measureVariable, measureFilter, boxColor, sizesAndStrides{1:end});

    net.mode = mode;
    net.conserveMemory = conserveMemory;
end

function heatmap = occlude(net, inputName, im_, score, measureVariable, measureFilter, boxColor, varargin)

    % pre-allocate the heatmap
    heatmap = zeros(size(im_,1), size(im_,2), 'double') ;

    % fill the heatmap
    for k = 1:4:length(varargin)

        tempHeat = compute_heat(net, inputName, im_, score, measureVariable, measureFilter, boxColor, varargin{k}, varargin{k+1}, varargin{k+2}, varargin{k+3});

        greatest = max(tempHeat(:));
        tempHeat = tempHeat ./ greatest;
        heatmap(:) = heatmap(:) + tempHeat(:);
    end

    greatest = max(heatmap(:));
    heatmap = heatmap ./ greatest;
end

function heatmap = compute_heat(net, inputName, im_, score, measureVariable, measureFilter, boxColor, sizeW, strideW, sizeH, strideH)

    gpuMode = isa(im_, 'gpuArray') ;

    % pre-allocate the heatmap
    if gpuMode
        heatmap = zeros(size(im_,1), size(im_,2), 'double', 'gpuArray') ;
    else
        heatmap = zeros(size(im_,1), size(im_,2), 'double') ;
    end

    randomColor = length(boxColor) ~= 3;

    % calculate the start and end points depending on the occlusion box
    % size and stride rate. Start and end points will either be the first
    % and last pixel of the image or outside the image area
    startW = 1 - sizeW + strideW;
    endW = size(im_,2) - strideW+1;
    startH = 1 - sizeH + strideH;
    endH = size(im_,1) - strideH + 1;

    range_d = 1:size(im_,3);

    fprintf('row: %3d, column: %3d', 0, 0);

    for w = startW:strideW:endW

        % cut off the left and right parts of the occlusion box that are
        % not inside the image area
        range_w = max(w,1):min(w+strideW-1,size(im_,2));

        for h = startH:strideH:endH

            % cut off the upper and lower parts of the occlusion box that
            % are not inside the image area
            range_h = max(h,1):min(h+strideH-1,size(im_,1));

            % copy image
            im2 = im_ ;

            %  and overlay the occlusion box over the copied image
            if randomColor
                im2(range_h, range_w, :) = rand(length(range_h), length(range_w), size(im2,3), 'single') * 256 - 128;
            else
                im2(range_h, range_w, 1) = boxColor(1);
                im2(range_h, range_w, 2) = boxColor(2);
                im2(range_h, range_w, 3) = boxColor(3);
            end

            % calculate activations (disable dropout)
            net.eval({inputName, im2});

            % add the norm of the difference in activations to the area of the
            % heatmap which corresponds to the occluded parts of the image
            scores = gather(net.vars(measureVariable).value);
            heatmap(range_h, range_w) = heatmap(range_h, range_w) + norm(score - squeeze(scores(:,:,measureFilter)));
            % heat(range_h, range_w) = heat(range_h, range_w) + (score - scores(filter));
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\brow: %3d, column: %3d', h, w);
        end
    end

    fprintf('\n');

    heatmap = gather(heatmap);
end

function measureFilter = getFilter(scores)
    % pre-allocate enough space for the activations of all filters
    tempScores = zeros(1, size(scores,3), 'double');

    % go through all filters and calculate the norm of the activation
    for i = 1:size(scores,3)
        tempScores(i) = norm(squeeze(scores(:,:,i)));
    end
    % assign the filter with the maximum activation
    [~, measureFilter] = max(tempScores);
end
