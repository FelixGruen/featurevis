function im_ = simplenn_reconstruction(net, im_, target, varargin)
%SIMPLENN_RECONSTRUCTION. Generating a reconstruction of the given target.
%   HEATMAP = SIMPLENN_RECONSTRUCTION(NET, IM_, TARGET)
%   generates an input image which leads to activations equal to TARGET in the
%   last layer of the network. The reconstruction can be controlled by optional
%   regularizers.
%   NET. The CNN to visualize.
%   IM_. The image form which to start the reconstruction, e.g. a zero or noise
%       image.
%   TARGET. The target of the reconstruction. A multi-dimensional array representing
%       the values of the activation maps of the last layer. The reconstructed
%       image will result in activations which are equal or close to the target.
%
%   SIMPLENN_OCCLUSION(...,'OPT',VALUE,...) takes the following options:
%
%   'Runs':: 100
%       An integer specifying the number of runs. One forward and the subsequent
%       backpropagation pass form one run.
%
%   'p':: 6
%       The p value for the p-norm. (Actually it's the p-norm to the power of p.)
%
%   'pnormFactor':: 1 / (size(im_,1) * size(im_,2) * 128 ^ p)
%       The factor with which to multiply the gradients of the p-norm.
%
%   'tvnormfactor':: 1 / (size(im_,1) * size(im_,2) * 128 ^ 2 * 0.01 ^ 2)
%       The factor with which to multiply the gradients of the total variation
%       regularizer.
%
%   'targetfactor':: 1 / (sum(target(:) .^ 2))
%       The factor with which to multiply the gradients of the l2 loss function.
%
%   'momentum':: 0.9
%       The value of the momentum. Set zero for no momentum.
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
    p = 6;
    runs = 100;
    pnormFactor = 1 / (size(im_,1) * size(im_,2) * 128 ^ p);
    tvnormFactor = 1 / (size(im_,1) * size(im_,2) * 128 ^ 2 * 0.01 ^ 2);
    targetFactor = 1 / (sum(target(:) .^ 2));
    momentum = 0.9;

    % parse optional parameters
    for i = 1:2:length(varargin)

        if ~ischar(varargin{i})
            error('The %d. parameter name is not of type char', fix(i/2));
        end

        switch lower(varargin{i})
            case 'runs'
                if ~isnumeric(varargin{i+1}) error('The value for runs must be an integer'); end
                runs = cast(varargin{i+1}, 'int32');
            case 'p'
                if ~isnumeric(varargin{i+1}) error('The value for p must be an integer'); end
                p = cast(varargin{i+1}, 'int32');
            case 'pnormfactor'
                if ~isnumeric(varargin{i+1}) error('The value for pnormFactor must be numeric'); end
                pnormFactor = varargin{i+1};
            case 'tvnormfactor'
                if ~isnumeric(varargin{i+1}) error('The value for tvnormFactor must be numeric'); end
                tvnormFactor = varargin{i+1};
            case 'targetfactor'
                if ~isnumeric(varargin{i+1}) error('The value for targetFactor must be numeric'); end
                targetFactor = varargin{i+1};
            case 'momentum'
                if ~isnumeric(varargin{i+1}) error('The value for momentum must be numeric'); end
                momentum = varargin{i+1};
            otherwise
                error('Unknown parameter "%s"', varargin{i});
        end
    end

    % --- preparations ---

    gpuMode = isa(im_, 'gpuArray');

    % move everything to the GPU
    if gpuMode
        net = vl_simplenn_move(net, 'gpu');
    else
        net = vl_simplenn_move(net, 'cpu');
    end

    % Display user information
    fprintf('Generating input image.\n');

    % --- generate the output ---

    rowDiff = zeros(size(im_,1), size(im_,2) - 1, size(im_,3), 'single');
    colDiff = zeros(size(im_,1) - 1, size(im_,2), size(im_,3), 'single');
    tvGrad = zeros(size(im_,1), size(im_,2), size(im_,3), 'single');
    sigma = zeros(size(im_,1), size(im_,2), size(im_,3), 'single');
    gradients = zeros(size(im_,1), size(im_,2), size(im_,3), 'single');

    if gpuMode
        target = gpuArray(target);
        rowDiff = gpuArray(rowDiff);
        colDiff = gpuArray(colDiff);
        tvGrad = gpuArray(tvGrad);
        sigma = gpuArray(sigma);
        gradients = gpuArray(gradients);
    end

    p1 = p-1;
    res = {};

    for i = 1:runs

        % calculate activations (disable dropout)
        res = vl_simplenn(net, im_, [], [], 'Mode', 'test');

        scores = res(end).x;

        % compute the derivatives of the activations
        dzdy = 2 * (scores - target) * targetFactor;

        res = vl_simplenn(net, im_, dzdy, res, 'Mode', 'test', 'SkipForward', true);

        % compute the derivatives of the total variation regularizer
        rowDiff(:,:,:) = im_(:, 2:end, :) - im_(:, 1:end-1, :);
        colDiff(:,:,:) = im_(2:end, :, :) - im_(1:end-1, :, :);
        tvGrad(2:end-1, 2:end-1, :) = rowDiff(2:end-1, 1:end-1, :) - rowDiff(2:end-1, 2:end, :) + colDiff(1:end-1, 2:end-1, :) - colDiff(2:end, 2:end-1, :);
        tvGrad(1, 2:end-1, :) = rowDiff(1, 1:end-1, :) - rowDiff(1, 2:end, :) - colDiff(1, 2:end-1, :);
        tvGrad(end, 2:end-1, :) = rowDiff(end, 1:end-1, :) - rowDiff(end, 2:end, :) + colDiff(end, 2:end-1, :);
        tvGrad(2:end-1, 1, :) = -rowDiff(2:end-1, 1, :) + colDiff(1:end-1, 1, :) - colDiff(2:end, 1, :);
        tvGrad(2:end-1, end, :) = rowDiff(2:end-1, end, :) + colDiff(1:end-1, end, :) - colDiff(2:end, end, :);
        tvGrad(1, 1, :) = -rowDiff(1, 1, :) - colDiff(1, 1, :);
        tvGrad(1, end, :) = rowDiff(1, end, :) - colDiff(1, end, :);
        tvGrad(end, 1, :) = -rowDiff(end, 1, :) + colDiff(end, 1, :);
        tvGrad(end, end, :) = rowDiff(end, end, :) + colDiff(end, end, :);

        gradients(:,:,:) = 2 * res(1).dzdx;

        % Add it all together
        % The derivatives of the pnorm are p * (sign(im_) .* (im_ .^ p1))
        gradients(:,:,:) = gradients + pnormFactor *  p * (sign(im_) .* (im_ .^ p1)) + tvnormFactor * 2 * tvGrad;

        % Use momentum
        sigma = momentum * sigma - gradients;

        im_ = im_ + sigma;

        % User output
        if mod(i,20) == 0 || i == 1
            loss = targetFactor * sum((scores(:) - target(:)) .^ 2);
            pnormLoss = pnormFactor * sum(im_(:) .^ p);
            tvLoss = tvnormFactor * (sum(rowDiff(:) .^ 2) + sum(colDiff(:) .^ 2));
            fprintf('run: %d, loss: %.3f = %.3f + %.3f + %.3f\n', i, loss+pnormLoss+tvLoss, loss, pnormLoss, tvLoss);
        end

    end
end
