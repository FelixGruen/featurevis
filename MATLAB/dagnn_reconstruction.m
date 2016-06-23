function im_ = dagnn_reconstruction(net, im_, target, varargin)
%DAGNN_RECONSTRUCTION. Generating a reconstruction of the given target.
%   IM_ = DAGNN_RECONSTRUCTION(NET, IM_, TARGET)
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
%   DAGNN_RECONSTRUCTION(...,'OPT',VALUE,...) takes the following options:
%
%   'InputName':: Empty
%      Sets the input variable. Only required for dag networks with more than
%      one input variable.
%
%   'OutputName':: Empty
%      Sets the output variable. Only required for dag networks with more than
%      one output variable.
%
%   'Runs':: 100
%       An integer specifying the number of runs. One forward and the subsequent
%       backpropagation pass form one run.
%
%   'p':: 6
%       The p value for the p-norm. (Actually it's the p-norm to the power of p.)
%
%   'pNormFactor':: 1 / (size(im_,1) * size(im_,2) * 128 ^ p)
%       The factor with which to multiply the gradients of the p-norm.
%
%   'tvNormFactor':: 1 / (size(im_,1) * size(im_,2) * 128 ^ 2 * 0.01 ^ 2)
%       The factor with which to multiply the gradients of the total variation
%       regularizer.
%
%   'targetFactor':: 1 / (sum(target(:) .^ 2))
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
    inputName = '';
    inputVariable = -1;
    outputName = '';
    outputVariable = -1;

    p = 6;
    runs = 100;
    pNormFactor = 1 / (size(im_,1) * size(im_,2) * 128 ^ p);
    tvNormFactor = 1 / (size(im_,1) * size(im_,2) * 128 ^ 2 * 0.01 ^ 2);
    targetFactor = 1 / (sum(target(:) .^ 2));
    momentum = 0.9;

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
            case 'runs'
                if ~isnumeric(varargin{i+1}) error('The value for runs must be an integer'); end
                runs = cast(varargin{i+1}, 'int32');
            case 'p'
                if ~isnumeric(varargin{i+1}) error('The value for p must be an integer'); end
                p = cast(varargin{i+1}, 'int32');
            case 'pnormfactor'
                if ~isnumeric(varargin{i+1}) error('The value for pNormFactor must be numeric'); end
                pNormFactor = varargin{i+1};
            case 'tvnormfactor'
                if ~isnumeric(varargin{i+1}) error('The value for tvNormFactor must be numeric'); end
                tvNormFactor = varargin{i+1};
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

    gpuMode = isa(im_, 'gpuArray');

    % move everything to the GPU
    if gpuMode
        net.move('gpu');
    else
        net.move('cpu');
    end

    % Disable dropout
    mode = net.mode;
    net.mode = 'test';

    fprintf('Input variable %d (%s). Output variable %d (%s)\n', inputVariable, inputName, outputVariable, outputName);

    % Display user information
    fprintf('Generating input image.\n');

    % --- generate the output ---

    sigma = zeros(size(im_,1), size(im_,2), size(im_,3), 'single');
    gradients = zeros(size(im_,1), size(im_,2), size(im_,3), 'single');

    if gpuMode
        target = gpuArray(target);
        sigma = gpuArray(sigma);
        gradients = gpuArray(gradients);
    end

    res = {};

    for i = 1:runs

        % calculate activations
        net.eval({inputName, im_});

        scores = net.vars(outputVariable).value;

        % compute the derivatives of the activations
        dzdy = 2 * (scores - target);

        net.eval({inputName, im_}, {outputName, dzdy});

        % Add all the derivatives together
        gradients(:,:,:) = targetFactor * net.vars(inputVariable).der;

        if pNormFactor ~= 0
            gradients(:,:,:) = gradients(:,:,:) + pNormFactor * pNormGradient(p, im_);
        end

        if tvNormFactor ~= 0
            gradients(:,:,:) = gradients(:,:,:) + tvNormFactor * tvNormGradient(im_, gpuMode);
        end

        % Use momentum
        sigma = momentum * sigma - gradients;

        im_ = im_ + sigma;

        % User output
        if mod(i,20) == 0 || i == 1
            loss = targetFactor * sum((scores(:) - target(:)) .^ 2);
            pNormLoss = pNormFactor * sum(abs(im_(:)) .^ p);
            tvLoss = tvNormFactor * tvNorm(im_);
            fprintf('run: %d, loss: %.3f = %.3f (activation loss) + %.3f (p-norm loss) + %.3f (tv norm loss)\n', i, loss+pNormLoss+tvLoss, loss, pNormLoss, tvLoss);
        end

    end

    net.mode = mode;
end

function grad = pNormGradient(p, im_)
    grad = p * sign(im_) .* (abs(im_) .^ (p-1));
end

function grad = tvNormGradient(im_, gpuMode)

    rowDiff = zeros(size(im_,1), size(im_,2) - 1, size(im_,3), 'single');
    colDiff = zeros(size(im_,1) - 1, size(im_,2), size(im_,3), 'single');
    grad = zeros(size(im_,1), size(im_,2), size(im_,3), 'single');

    if gpuMode
        rowDiff = gpuArray(rowDiff);
        colDiff = gpuArray(colDiff);
        grad = gpuArray(grad);
    end

    rowDiff(:,:,:) = im_(:, 2:end, :) - im_(:, 1:end-1, :);
    colDiff(:,:,:) = im_(2:end, :, :) - im_(1:end-1, :, :);

    grad(2:end-1, 2:end-1, :) = rowDiff(2:end-1, 1:end-1, :) - rowDiff(2:end-1, 2:end, :) + colDiff(1:end-1, 2:end-1, :) - colDiff(2:end, 2:end-1, :);

    grad(1, 2:end-1, :) = rowDiff(1, 1:end-1, :) - rowDiff(1, 2:end, :) - colDiff(1, 2:end-1, :);
    grad(end, 2:end-1, :) = rowDiff(end, 1:end-1, :) - rowDiff(end, 2:end, :) + colDiff(end, 2:end-1, :);

    grad(2:end-1, 1, :) = -rowDiff(2:end-1, 1, :) + colDiff(1:end-1, 1, :) - colDiff(2:end, 1, :);
    grad(2:end-1, end, :) = rowDiff(2:end-1, end, :) + colDiff(1:end-1, end, :) - colDiff(2:end, end, :);

    grad(1, 1, :) = -rowDiff(1, 1, :) - colDiff(1, 1, :);
    grad(1, end, :) = rowDiff(1, end, :) - colDiff(1, end, :);
    grad(end, 1, :) = -rowDiff(end, 1, :) + colDiff(end, 1, :);
    grad(end, end, :) = rowDiff(end, end, :) + colDiff(end, end, :);
end

function loss = tvNorm(im_)

    im = gather(im_);
    loss = 0.0;

    for k = 1:size(im,3)

        for j = 1:(size(im,2)-1)
            for i = 1:(size(im,1)-1)
                loss = loss + (im(i,j+1,k) - im(i,j,k)) ^ 2 + (im(i+1,j,k) - im(i,j,k)) ^ 2;
            end
            loss = loss + (im(end,j+1,k) - im(end,j,k)) ^ 2; % edge case
        end

        for i = 1:(size(im,1)-1)
            loss = loss + (im(i+1,end,k) - im(i,end,k)) ^ 2; % edge case
        end
    end

end
