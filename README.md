# The FeatureVis library

## Quick Start Guide

## Functions

##### graphical_deconvolution(net, im, im_)

Deconvolves activations back to the input space to generate a heatmap of activations.

**net** The CNN to visualize.  
**im** The original input image. Needed for user output only.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.

**graphical_deconvolution(..., 'opt', value, ...)** takes the following options:  

**'InputName' :: Empty**  
Sets the input variable. Only required for DAG networks with more than one input variable.

**'OutputName' :: Empty**  
Sets the output variable. Only required for DAG networks with more than one output variable.

**'ReLUPass' :: 'Guided Backpropagation'**  
Sets the method used to deconvolve activations through the ReLU layers. The available methods are 'Backpropagation', 'Deconvnet', and 'Guided Backpropagation'. The default method is 'Guided Backpropagation' since this method usually gives the best results.

**'ConvolutionPass' :: 'Standard'**  
Sets the method used to deconvolve activations through the convolution layers. The available methods are 'Relevance Propagation', and 'Standard'. The default method is 'Standard', since this method usually gives the best results.

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer from which activations should be deconvolved back to the input layer. By default the last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter for which activations should be deconvolved back to the input layer. By default the strongest activated filter is used.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they
are run on the CPU.

Calls simplenn_graphical_deconvolution or dagnn_graphical_deconvolution internally.

##### simplenn_graphical_deconvolution(net, im, im_)

Ddeconvolves activations back to the input space to generate a heatmap of activations, and displays it to the user.

**net** The CNN to visualize.  
**im** The original input image. Needed for user output only.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.

**simplenn_graphical_deconvolution(..., 'opt', value, ...)** takes the following options:

**'ReLUPass' :: 'Guided Backpropagation'**  
Sets the method used to deconvolve activations through the ReLU layers. The available methods are 'Backpropagation', 'Deconvnet', and 'Guided Backpropagation'. The default method is 'Guided Backpropagation' since this method usually gives the best results.

**'ConvolutionPass' :: 'Standard'**  
Sets the method used to deconvolve activations through the convolution layers. The available methods are 'Relevance Propagation', and 'Standard'. The default method is 'Standard', since this method usually gives the best results.

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer from which activations should be deconvolved back to the input layer. By default the last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter for which activations should be deconvolved back to the input layer. By default the strongest activated filter is used.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.

Calls simplenn_deconvolution internally.

##### simplenn_deconvolution(net, im_)

Ddeconvolves activations back to the input space to generate a heatmap of activations, and returns the heatmap.

**net** The CNN to visualize.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.

**simplenn_deconvolution(..., 'opt', value, ...)** takes the following options:

**'ReLUPass' :: 'Guided Backpropagation'**  
Sets the method used to deconvolve activations through the ReLU layers. The available methods are 'Backpropagation', 'Deconvnet', and 'Guided Backpropagation'. The default method is 'Guided Backpropagation' since this method usually gives the best results.

**'ConvolutionPass' :: 'Standard'**  
Sets the method used to deconvolve activations through the convolution layers. The available methods are 'Relevance Propagation', and 'Standard'. The default method is 'Standard', since this method usually gives the best results.

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer from which activations should be deconvolved back to the input layer. By default the last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter for which activations should be deconvolved back to the input layer. By default the strongest activated filter is used.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.

##### dagnn_graphical_deconvolution(net, im, im_)

Ddeconvolves activations back to the input space to generate a heatmap of activations, and displays it to the user.

**net** The CNN to visualize.  
**im** The original input image. Needed for user output only.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.

**dagnn_graphical_deconvolution(..., 'opt', value, ...)** takes the following options:

**'InputName' :: Empty**  
Sets the input variable. Only required for DAG networks with more than one input variable.

**'OutputName' :: Empty**  
Sets the output variable. Only required for DAG networks with more than one output variable.

**'ReLUPass' :: 'Guided Backpropagation'**  
Sets the method used to deconvolve activations through the ReLU layers. The available methods are 'Backpropagation', 'Deconvnet', and 'Guided Backpropagation'. The default method is 'Guided Backpropagation' since this method usually gives the best results.

**'ConvolutionPass' :: 'Standard'**  
Sets the method used to deconvolve activations through the convolution layers. The available methods are 'Relevance Propagation', and 'Standard'. The default method is 'Standard', since this method usually gives the best results.

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer from which activations should be deconvolved back to the input layer. By default the last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter for which activations should be deconvolved back to the input layer. By default the strongest activated filter is used.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.

Calls dagnn_deconvolution internally.

##### dagnn_deconvolution(net, im, im_, inputName, outputName)

Ddeconvolves activations back to the input space to generate a heatmap of activations, and displays it to the user.

**net** The CNN to visualize.  
**im** The original input image. Needed for user output only.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.  
**inputName** The name of the input variable of the network.  
**outputName** The name of the output variable of the network.

**dagnn_deconvolution(..., 'opt', value, ...)** takes the following options:

**'ReLUPass' :: 'Guided Backpropagation'**  
Sets the method used to deconvolve activations through the ReLU layers. The available methods are 'Backpropagation', 'Deconvnet', and 'Guided Backpropagation'. The default method is 'Guided Backpropagation' since this method usually gives the best results.

**'ConvolutionPass' :: 'Standard'**  
Sets the method used to deconvolve activations through the convolution layers. The available methods are 'Relevance Propagation', and 'Standard'. The default method is 'Standard', since this method usually gives the best results.

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer from which activations should be deconvolved back to the input layer. By default the last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter for which activations should be deconvolved back to the input layer. By default the strongest activated filter is used.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.
