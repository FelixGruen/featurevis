function heatmap = dagnn_deconvolution(net, im_, inputName, outputName, varargin)
%DAGNN_DECONVOLUTION. Deconvoluting activations back to the input layer.
%   HEATMAP = DAGNN_DECONVOLUTION(NET, IM_, INPUTNAME, OUTPUTNAME)
%   deconvolves activations to generate a heatmap of activations,
%   and returns the heatmap.

%   NET. The CNN of type DDagNN to visualize.
%   IM_. The adjusted input image used as input to the CNN. If IM_ is a
%        gpuArray computations are run on the GPU.
%   INPUTNAME. The name of the input variable of the network.
%   OUTPUTNAME. The name of the output variable of the network.
%
%   DAGNN_DECONVOLUTION(...,'OPT',VALUE,...) takes the following options:
%
%   'ReLUPass':: 'Guided Backpropagation'
%       Sets the method used to deconvolve activations through the ReLU layers.
%       The available methods are 'Backpropagation', 'Deconvnet', and 'Guided
%       Backpropagation'. The default method is 'Guided Backpropagation'
%       since this method usually gives the best results.
%
%   'ConvolutionPass':: 'Standard'
%       Sets the method used to deconvolve activations through the convolution
%       layers. The available methods are 'Relevance Propagation', and
%       'Standard'. The default method is 'Standard', since this
%       method usually gives the best results.
%
%   'MeasureLayer':: Last layer (not SoftMax)
%       An Int32 specifying the layer from which activations should be
%       deconvoluted back to the input layer. By default the last layer of
%       the network is used, unless the last layer is a SoftMax layer, in
%       which case the second to last layer is used.
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
    reluMethod = 'guided backpropagation';
    convMethod = 'standard';
    measureLayer = 0;
    measureVariable = 0;
    measureFilter = 0;

    % parse optional parameters
    for i = 1:2:length(varargin)

        if ~ischar(varargin{i})
            error('The %d. parameter name is not of type char', fix(i/2));
        end

        switch lower(varargin{i})
        case 'relupass'
            if ~ischar(varargin{i+1}) error('The value for ReLUPass must be a char'); end
            reluMethod = lower(varargin{i+1});
            if ~strcmp(reluMethod, 'backpropagation') && ~strcmp(reluMethod, 'deconvnet') && ~strcmp(reluMethod, 'guided backpropagation')
                error('Unknown value for parameter "ReLUPass": "%s"', reluMethod);
            end
        case 'convolutionpass'
            if ~ischar(varargin{i+1}) error('The value for ConvolutionPass must be a char'); end
            convMethod = lower(varargin{i+1});
            if ~strcmp(convMethod, 'relevance propagation') && ~strcmp(convMethod, 'standard')
                error('Unknown value for parameter "ConvolutionPass": "%s"', convMethod);
            end
        case 'measurelayer'
            if ~isnumeric(varargin{i+1}) error('The value for measureLayer must be an integer'); end
            measureLayer = varargin{i+1};
            if measureLayer > numel(net.layers)
                error('Measure layer (%d) cannot be greater than number of layers in the network (%d)', measureLayer, numel(net.layers));
            end
            measureVariable = net.getVarIndex(net.layers(measureLayer).outputs(1));
        case 'measurefilter'
            if ~isnumeric(varargin{i+1}) error('The value for measureFilter must be an integer'); end
            measureFilter = cast(varargin{i+1}, 'int32');
        otherwise
                error('Unknown parameter "%s"', varargin{i});
        end
    end

    % if the measureLayer was not set by the user set it automatically
    if measureLayer < 1

        % find the output layer
        for i = length(net.layers):-1:1
            if ismember(outputVariable, net.layers(i).outputIndexes)
                break;
            end
        end

        % if the output layer is a softmax layer find the previous layer
        if isa(net.layers(i).block, 'dagnn.SoftMax')

            in = net.layers(i).inputIndexes; % inputIndexes is just one value

            for i = length(net.layers):-1:1
                if ismember(in, net.layers(i).outputIndexes)
                    measureLayer = i;
                    measureVariable = in;
                    break;
                end
            end
        % otherwise use the output layer
        else
            measureLayer = i;
            measureVariable = outputVariable;
        end
    end

    if measureFilter < 0
        error('Measure filter (%d) cannot be negativ', measureFilter);
    end

    % --- preparations ---

    % Result layer one will be the input, so all other layers must be moved up one
    % measureLayer = measureLayer + 1;

    gpuMode = isa(im_, 'gpuArray') ;

    % move net to the GPU
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

    % forward pass
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
    fprintf('Deconvoluting activations of filter %d of layer %d (%s) using %s for the pass through ReLUs and\nthe %s method to pass through the convolutional layers.\n', ...
        measureFilter, measureLayer, net.layers(measureLayer).name, reluMethod, convMethod);

    % only keep the filter of interest. set all others to zero
    dzdy = zeros(size(scores,1), size(scores,2), size(scores,3), 'single');
    if gpuMode
        dzdy = gpuArray(dzdy);
    end
    dzdy(:,:,measureFilter) = scores(:,:,measureFilter);

    % --- compute deconvolution ---

    % set output derivatives
    net.deconv(dzdy, measureVariable, measureLayer, reluMethod, convMethod);

    heatmap = gather(net.vars(inputVariable).der);

    net.mode = mode;
    net.conserveMemory = conserveMemory;
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
