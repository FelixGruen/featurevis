classdef DDagNN < dagnn.DagNN

% Copyright (C) 2016 Felix Grün.
% All rights reserved.
%
% Parts of the code taken and modified from the MatConvNet library made available
% under the terms of the BSD license (see the MATCONVNET_LICENCE file).
% Copyright (C) 2015-2016 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the FeatureVis library and is made available under
% the terms of the BSD license (see the LICENCE file).

  properties (Transient, Access = private, Hidden = true)
    modifed = false
    varNames = struct()
    paramNames = struct()
    layerNames = struct()
    layerIndexes = {}
  end

  methods
    function obj = DDagNN()
      obj.vars = struct(...
        'name', {}, ...
        'value', {}, ...
        'der', {}, ...
        'fanin', {}, ...
        'fanout', {}, ...
        'precious', {}) ;
      obj.params = struct(...
        'name', {}, ...
        'value', {}, ...
        'der', {}, ...
        'fanout', {}, ...
        'trainMethod', {}, ...
        'learningRate', {}, ...
        'weightDecay', {}) ;
      obj.layers = struct(...
        'name', {}, ...
        'inputs', {}, ...
        'outputs', {}, ...
        'params', {}, ...
        'inputIndexes', {}, ...
        'outputIndexes', {}, ...
        'paramIndexes', {}, ...
        'forwardTime', {[]}, ...
        'backwardTime', {[]}, ...
        'block', {}) ;
    end

    % Manage the DagNN
    reset(obj)
    function move(obj, direction)
        move@dagnn.DagNN(obj, direction);
    end

    % Manipualte the DagNN
    addLayer(obj, name, block, inputs, outputs, params)
    removeLayer(obj, name)
    setLayerInputs(obj, leyer, inputs)
    setLayerOutput(obj, layer, outputs)
    setLayerParams(obj, layer, params)
    renameVar(obj, oldName, newName, varargin)
    rebuild(obj)

    % Process data with the DagNN
    initParams(obj)
    eval(obj, inputs, derOutputs)
    eval_occlusion(obj, inputs, occlusionVariable, occlusionFilter, range_h, range_w)
    deconv(net, dzdy, outputIndex, outputLayer, reluMethod, convMethod)

    % Get information about the DagNN
    varSizes = getVarSizes(obj, inputSizes)

    % ---------------------------------------------------------------------
    %                                                           Access data
    % ---------------------------------------------------------------------

    function inputs = getInputs(obj)
        inputs = getInputs@dagnn.DagNN(obj);
    end

    function outputs = getOutputs(obj)
        outputs = getOutputs@dagnn.DagNN(obj);
    end

    function l = getLayerIndex(obj, name)
        l = getLayerIndex@dagnn.DagNN(obj, name);
    end

    function v = getVarIndex(obj, name)
        v = getVarIndex@dagnn.DagNN(obj, name);
    end

    function p = getParamIndex(obj, name)
        p = getParamIndex@dagnn.DagNN(obj, name);
    end

    function layer = getLayer(obj, layerName)
        layer = getLayer@dagnn.DagNN(obj, layerName);
    end

    function var = getVar(obj, varName)
        var = getVar@dagnn.DagNN(obj, varName);
    end

    function param = getParam(obj, paramName)
        param = getParam@dagnn.DagNN(obj, paramName);
    end

    function order = getLayerExecutionOrder(obj)
        order = getLayerExecutionOrder@dagnn.DagNN(obj);
    end
  end

  methods (Static)
    obj = fromSimpleNN(net, varargin)
    obj = loadobj(s)
  end

  methods (Access = {?dagnn.DagNN, ?dagnn.Layer})
    function v = addVar(obj, name)
        v = addVar@dagnn.DagNN(obj, name);
    end

    function p = addParam(obj, name)
        p = addParam@dagnn.DagNN(obj, name);
    end
  end


    %{
    properties (Transient, Access = {?dagnn.DagNN, ?dagnn.Layer}, Hidden = true)
        numPendingVarRefs
        numPendingParamRefs
        computingDerivative = false
        executionOrder
    end

    properties (Transient, Access = private, Hidden = true)
        modifed = false
        varNames = struct()
        paramNames = struct()
        layerNames = struct()
        layerIndexes = {}
    end

    methods

        % Manage the DagNN
        function move(obj, direction)
            move@dagnn.DagNN(obj, direction);
        end

        % Process data with the DagNN
        initParams(obj)

        function eval(obj, inputs, derOutputs)
            disp(obj.vars(1).name);
            if ~exist('derOutputs', 'var')
                eval@dagnn.DagNN(obj, inputs);
            else
                eval@dagnn.DagNN(obj, inputs, derOutputs);
            end
        end

        % Get information about the DagNN
        function varSizes = getVarSizes(obj, inputSizes)
            varSizes = getVarSizes@dagnn.DagNN(obj, inputSizes);
        end

        % ---------------------------------------------------------------------
        %                                                           Access data
        % ---------------------------------------------------------------------

        function inputs = getInputs(obj)
            inputs = getInputs@dagnn.DagNN(obj);
        end

        function outputs = getOutputs(obj)
            outputs = getOutputs@dagnn.DagNN(obj);
        end

        function l = getLayerIndex(obj, name)
            l = getLayerIndex@dagnn.DagNN(obj, name);
        end

        function v = getVarIndex(obj, name)
            disp('here');
            obj.
            v = getVarIndex@dagnn.DagNN(obj, name);
        end

        function p = getParamIndex(obj, name)
            p = getParamIndex@dagnn.DagNN(obj, name);
        end
    end
    %}
end
