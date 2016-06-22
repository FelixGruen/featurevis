function heatmap = graphical_deconvolution(net, im, im_, varargin)
%GRAPHICAL_DECONVOLUTION. Deconvoluting activations back to the input layer.
%   HEATMAP = GRAPHICAL_DECONVOLUTION(NET, IM, IM_) deconvolutes activations
%   to generate a heatmap of activations.
%   NET. The CNN to visualize.
%   IM. The original input image. Needed for user output only.
%   IM_. The adjusted input image used as input to the CNN. If IM_ is a
%        gpuArray computations are run on the GPU.
%
%   GRAPHICAL_DECONVOLUTION(...,'OPT',VALUE,...) takes the following options:
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


    if isa(net, 'DDagNN') || isa(net, 'dagnn.DagNN')
        heatmap = dagnn_graphical_deconvolution(net, im, im_, varargin{:});
    else
        heatmap = simplenn_graphical_deconvolution(net, im, im_, varargin{:});
    end
end
