function heatmap = graphical_occlusion(net, im, im_, varargin)
%GRAPHICAL_OCCLUSION. Generating a heatmap of feature importance.
%   HEATMAP = GRAPHICAL_OCCLUSION(NET, IM, IM_, {SIZE, STRIDE})
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
%   GRAPHICAL_OCCLUSION(...,'OPT',VALUE,..., {SIZE, STRIDE}) takes the following options:
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


    if isa(net, 'DDagNN') || isa(net, 'dagnn.DagNN')
        heatmap = dagnn_graphical_occlusion(net, im, im_, varargin{:});
    else
        heatmap = simplenn_graphical_occlusion(net, im, im_, varargin{:});
    end
end
