function deconv(net, dzdy, outputIndex, outputLayer, reluMethod, convMethod)

% Copyright (C) 2016 Felix Grün.
% All rights reserved.
%
% Parts of the code taken and modified from the MatConvNet library made available
% under the terms of the BSD license (see the MATCONVNET_LICENCE file).
% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the FeatureVis library and is made available under
% the terms of the BSD license (see the LICENCE file).


    % set output derivatives
    net.vars(outputIndex).der = dzdy ;
    net.numPendingVarRefs = zeros(1, numel(net.vars)) ;
    net.numPendingParamRefs = zeros(1, numel(net.params)) ;

    started = false;

    for l = fliplr(net.executionOrder)
        if ~started && l == outputLayer
            started = true;
        end

        if started
            time = tic ;
            if isa(net.layers(l).block, 'dagnn.ReLU')
                backwardAdvancedReLU(net.layers(l), reluMethod);
            elseif isa(net.layers(l).block, 'dagnn.Conv')
                if l == length(net.layers)
                    backwardAdvancedConv(net.layers(l), net.layers(l), convMethod);
                else
                    backwardAdvancedConv(net.layers(l), net.layers(l+1), convMethod);
                end
            else
                net.layers(l).block.backwardAdvanced(net.layers(l));
            end
            net.layers(l).backwardTime = toc(time);
        end
    end
end

function backwardAdvancedConv(layer, nextLayer, method)
  in = layer.inputIndexes ;
  out = layer.outputIndexes ;
  par = layer.paramIndexes ;
  if strcmp(layer.name, nextLayer.name) == 1
    nextIn = layer.outputIndexes;
  else
    nextIn = nextLayer.inputIndexes;
  end
  net = layer.block.net ;

  nextInputs = {net.vars(nextIn).value};
  inputs = {net.vars(in).value} ;
  derOutputs = {net.vars(out).der} ;
  for i = 1:numel(derOutputs)
    if isempty(derOutputs{i}), return ; end
  end

  % compute derivatives of inputs and paramerters
  [derInputs, derParams] = backwardConv(layer.block, inputs, {net.params(par).value}, derOutputs, nextInputs, method) ;

  % accumuate derivatives
  for i = 1:numel(in)
    v = in(i) ;
    if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
      net.vars(v).der = derInputs{i} ;
    elseif ~isempty(derInputs{i})
      net.vars(v).der = net.vars(v).der + derInputs{i} ;
    end
    net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
  end

  for i = 1:numel(par)
    p = par(i) ;
    if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
          || isempty(net.params(p).der)
      net.params(p).der = derParams{i} ;
    else
      net.params(p).der = net.params(p).der + derParams{i} ;
    end
    net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1 ;
  end
end

function [derInputs, derParams] = backwardConv(obj, inputs, params, derOutputs, nextInputs, method)
    if ~obj.hasBias, params{2} = [] ; end

    if strcmp(method, 'relevance propagation');
        eps = 0.001 * ((nextInputs{1} > 0) - (nextInputs{1} < 0));
        dzdy = derOutputs{1} ./ (nextInputs{1} + eps);
    else
        dzdy = derOutputs{1};
    end

    [dzdx, derParams{1}, derParams{2}] = vl_nnconv(...
        inputs{1}, params{1}, params{2}, dzdy, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;

    if strcmp(method, 'relevance propagation');
        derInputs{1} = dzdx .* inputs{1};
    else
        derInputs{1} = dzdx;
    end
end

function backwardAdvancedReLU(layer, method)
    in  = layer.inputIndexes ;
    out = layer.outputIndexes ;
    par = layer.paramIndexes ;
    net = layer.block.net ;

    inputs     = {net.vars(in).value} ;
    derOutputs = {net.vars(out).der} ;

    for i = 1:numel(derOutputs)
        if isempty(derOutputs{i}), return ; end
    end

    % compute derivatives of inputs and paramerters
    [derInputs, derParams] = backwardReLU(inputs, derOutputs, method) ;

    % accumuate derivatives
    for i = 1:numel(in)
        v = in(i) ;
        if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
            net.vars(v).der = derInputs{i} ;
        elseif ~isempty(derInputs{i})
            net.vars(v).der = net.vars(v).der + derInputs{i} ;
        end
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
    end

    for i = 1:numel(par)
        p = par(i) ;
        if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
                || isempty(net.params(p).der)
            net.params(p).der = derParams{i} ;
        else
            net.params(p).der = net.params(p).der + derParams{i} ;
        end
        net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1 ;
    end
end

function [derInputs, derParams] = backwardReLU(inputs, derOutputs, method)
    x = inputs{1} ;
    dzdy = derOutputs{1} ;

    switch method
        case 'backpropagation'
            y = dzdy .* (x > 0) ;
        case 'deconvnet'
            y = dzdy .* (dzdy > 0) ;
        case 'guided backpropagation'
            y = dzdy .* (x > 0) .* (dzdy > 0) ;
    end

    derInputs{1} = y ;
    derParams = {} ;
end
