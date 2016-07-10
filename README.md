# The FeatureVis library

## Content

1. [Quick Start Guide] (#1-quick-start-guide)
2. [Functions] (#2-functions)
3. [Citation] (#3-citation)
4. [License] (#4-license)

## 1. Quick Start Guide

[[Content] (#content)] [Quick Start Guide] [[Functions] (#2-functions)] [[Citation] (#3-citation)] [[License] (#4-license)]

Every function requires a network and an image. So let's load the network first. After you have loaded MatConvNet use the normal load function for networks of type simpleNN:

```Matlab
net = load('your-simplenn-network.mat');
```

For networks of type DagNN use:

```Matlab
net = DDagNN.loadobj(load('your-dag-network.mat'));
```

A function shipped with the FeatureVis library. The DDagNN object implements special functions needed for the deconvolutional methods.

Next we need an image, once in its original version and once resized and normalized according to the specifications of the network you are going to use:

```Matlab
im = imread('your-image.jpg');
im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;
```

Now we can do some visualizations. Let's start by using Guided Backpropagation on the strongest activated class:

```Matlab
graphical_deconvolution(net, im, im_);
```

Let's compare it to normal Backpropagation:

```Matlab
graphical_deconvolution(net, im, im_, 'ReLUPass', 'Backpropagation');
```

Maybe we are interested in the features supporting another class:

```Matlab
graphical_deconvolution(net, im, im_, 'MeasureFilter', 42);
```

What if we want the computations to run on the GPU? Just use a gpuArray as input:

```Matlab
im_ = gpuArray(im_);
graphical_deconvolution(net, im, im_);
```

This works for all methods.

Let's use the occlusion method. For this method we have to specify the side length of the occluded area and the step width with which the occlusion box is moved across the image. Let's use a size of 32x32 pixels and a step width of 16 pixels:

```Matlab
graphical_occlusion(net, im, im_, {32, 16});
```

Maybe we want our occlusion box to have a specific color. Then we can specify it as RGB:

```Matlab
graphical_occlusion(net, im, im_, 'BoxColor', [128 64 -128], {32, 16});
```

Pay attention to the fact that these colors are used unmodified. So use the normalized image as basis for your color space.

What if we want a rectangular occlusion box? Just replace the two values from above with four to specify the size and stride along the width and the size and stride along the height of the image, so for a occlusion box 64 pixels wide and 32 pixels high use:

```Matlab
graphical_occlusion(net, im, im_, {64, 16, 32, 16});
```

To move the occlusion box with a step width of 16 pixels along the width and 32 pixels along the height of the image use:

```Matlab
graphical_occlusion(net, im, im_, {64, 16, 64, 32});
```

This concludes this short introduction. We hope you have fun experimenting with this library. The documentation for all methods can be found below or in the source code.

## 2. Functions

[[Content] (#content)] [[Quick Start Guide] (#1-quick-start-guide)] [Functions] [[Citation] (#3-citation)] [[License] (#4-license)]

#### graphical_deconvolution(net, im, im_)

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

#### simplenn_graphical_deconvolution(net, im, im_)

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

#### heatmap = simplenn_deconvolution(net, im_)

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

#### dagnn_graphical_deconvolution(net, im, im_)

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

#### heatmap = dagnn_deconvolution(net, im_, inputName, outputName)

Ddeconvolves activations back to the input space to generate a heatmap of activations, and displays it to the user.

**net** The CNN to visualize.  
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

#### graphical_occlusion(net, im, im_, {size, stride})

Generates a heatmap showing the importance of different areas of the input image, and displays it to the user.

**net** The CNN to visualize.  
**im** The original input image. Needed for user output only.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.  
**size** The size (side length) of the occluded area.  
**stride** The stride width with which the occluded area will be across the occluded filter.

**graphical_occlusion(..., 'opt', value, ..., {size, stride})** takes the following options:

**'InputName' :: Empty**  
Sets the input variable. Only required for DAG networks with more than one input variable.

**'OutputName' :: Empty**  
Sets the output variable. Only required for DAG networks with more than one output variable.

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer at which the changes in activations should be measured. By default the last layer or, if the last layer is a softmax layer, the second to last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter at which changes in activations should be measured. By default the strongest activated filter is used.

**'BoxColor' ::  Random pixel values**  
A 1x3 single Array specifying the color to be used for the occlusion box. The three values correspond to the three color channels.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.

Size and stride of the occluded area are always specified last. The two values for size and stride may be replaced by four values where the first two values specify the size and stride along the width of the input (horizontally) and the last two values specify the size and stride along the height of the input (vertically). In fact any number of cell arrays of two or four values may be used to run more than one test and average the results, e.g. two cell arrays might specify a first test with a larger occluded area and stride rate and a second test with a smaller occluded area and stride rate last. The resulting heatmaps will be normalized before being added together to form the final heatmap of feature importance.

Calls simplenn_graphical_occlusion or dagnn_graphical_occlusion internally.

#### simplenn_graphical_occlusion(net, im, im_, {size, stride})

Generates a heatmap showing the importance of different areas of the input image, and displays it to the user.

**net** The CNN to visualize.  
**im** The original input image. Needed for user output only.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.  
**size** The size (side length) of the occluded area.  
**stride** The stride width with which the occluded area will be across the occluded filter.

**simplenn_graphical_occlusion(..., 'opt', value, ..., {size, stride})** takes the following options:

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer at which the changes in activations should be measured. By default the last layer or, if the last layer is a softmax layer, the second to last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter at which changes in activations should be measured. By default the strongest activated filter is used.

**'BoxColor' ::  Random pixel values**  
A 1x3 single Array specifying the color to be used for the occlusion box. The three values correspond to the three color channels.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.

Size and stride of the occluded area are always specified last. The two values for size and stride may be replaced by four values where the first two values specify the size and stride along the width of the input (horizontally) and the last two values specify the size and stride along the height of the input (vertically). In fact any number of cell arrays of two or four values may be used to run more than one test and average the results, e.g. two cell arrays might specify a first test with a larger occluded area and stride rate and a second test with a smaller occluded area and stride rate last. The resulting heatmaps will be normalized before being added together to form the final heatmap of feature importance.

Calls simplenn_occlusion internally.

#### heatmap = simplenn_occlusion(net, im_, {size, stride})

Generates a heatmap showing the importance of different areas of the input image, and returns the heatmap.

**net** The CNN to visualize.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.  
**size** The size (side length) of the occluded area.  
**stride** The stride width with which the occluded area will be across the occluded filter.

**simplenn_occlusion(..., 'opt', value, ..., {size, stride})** takes the following options:

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer at which the changes in activations should be measured. By default the last layer or, if the last layer is a softmax layer, the second to last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter at which changes in activations should be measured. By default the strongest activated filter is used.

**'BoxColor' ::  Random pixel values**  
A 1x3 single Array specifying the color to be used for the occlusion box. The three values correspond to the three color channels.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.

Size and stride of the occluded area are always specified last. The two values for size and stride may be replaced by four values where the first two values specify the size and stride along the width of the input (horizontally) and the last two values specify the size and stride along the height of the input (vertically). In fact any number of cell arrays of two or four values may be used to run more than one test and average the results, e.g. two cell arrays might specify a first test with a larger occluded area and stride rate and a second test with a smaller occluded area and stride rate last. The resulting heatmaps will be normalized before being added together to form the final heatmap of feature importance.

#### dagnn_graphical_occlusion(net, im, im_, {size, stride})

Generates a heatmap showing the importance of different areas of the input image, and displays it to the user.

**net** The CNN to visualize.  
**im** The original input image. Needed for user output only.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.  
**size** The size (side length) of the occluded area.  
**stride** The stride width with which the occluded area will be across the occluded filter.

**dagnn_graphical_occlusion(..., 'opt', value, ..., {size, stride})** takes the following options:

**'InputName' :: Empty**  
Sets the input variable. Only required for DAG networks with more than one input variable.

**'OutputName' :: Empty**  
Sets the output variable. Only required for DAG networks with more than one output variable.

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer at which the changes in activations should be measured. By default the last layer or, if the last layer is a softmax layer, the second to last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter at which changes in activations should be measured. By default the strongest activated filter is used.

**'BoxColor' ::  Random pixel values**  
A 1x3 single Array specifying the color to be used for the occlusion box. The three values correspond to the three color channels.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.

Size and stride of the occluded area are always specified last. The two values for size and stride may be replaced by four values where the first two values specify the size and stride along the width of the input (horizontally) and the last two values specify the size and stride along the height of the input (vertically). In fact any number of cell arrays of two or four values may be used to run more than one test and average the results, e.g. two cell arrays might specify a first test with a larger occluded area and stride rate and a second test with a smaller occluded area and stride rate last. The resulting heatmaps will be normalized before being added together to form the final heatmap of feature importance.

Calls dagnn_occlusion internally.

#### heatmap = dagnn_occlusion(net, im, im_, inputName, outputName, {size, stride})

Generates a heatmap showing the importance of different areas of the input image, and returns the heatmap.

**net** The CNN to visualize.  
**im** The original input image. Needed for user output only.  
**im_** The adjusted input image used as input to the CNN. If im_ is a gpuArray computations are run on the GPU.
**inputName** The name of the input variable of the network.  
**outputName** The name of the output variable of the network.   
**size** The size (side length) of the occluded area.  
**stride** The stride width with which the occluded area will be across the occluded filter.

**dagnn_occlusion(..., 'opt', value, ..., {size, stride})** takes the following options:

**'MeasureLayer' :: Last layer**  
An Int32 specifying the layer at which the changes in activations should be measured. By default the last layer or, if the last layer is a softmax layer, the second to last layer of the network is used.

**'MeasureFilter' :: Strongest activated filter**  
An Int32 specifying the filter at which changes in activations should be measured. By default the strongest activated filter is used.

**'BoxColor' ::  Random pixel values**  
A 1x3 single Array specifying the color to be used for the occlusion box. The three values correspond to the three color channels.

**Comments**

Computations are run on the GPU if im_ is a gpuArray. Otherwise they are run on the CPU.

Size and stride of the occluded area are always specified last. The two values for size and stride may be replaced by four values where the first two values specify the size and stride along the width of the input (horizontally) and the last two values specify the size and stride along the height of the input (vertically). In fact any number of cell arrays of two or four values may be used to run more than one test and average the results, e.g. two cell arrays might specify a first test with a larger occluded area and stride rate and a second test with a smaller occluded area and stride rate last. The resulting heatmaps will be normalized before being added together to form the final heatmap of feature importance.

#### im_ = simplenn_reconstruction(net, im_, target)

Generates an input image which leads to activations equal to target in the last layer of the network. The reconstruction can be controlled by optional regularizers.

**net** The CNN to visualize.
**im_** The image form which to start the reconstruction, e.g. a zero or noise image.
**target** The target of the reconstruction. A multi-dimensional array representing the values of the activation maps of the last layer. The reconstructed image will result in activations which are equal or close to the target.

**simplenn_reconstruction(..., 'opt', value, ...)** takes the following options:

**'Runs' :: 100**  
An integer specifying the number of runs. One forward and the subsequent backpropagation pass form one run.

**'p' :: 6**  
The p value for the p-norm. (Actually it's the p-norm to the power of p.)

**'pNormFactor' :: 1 / (size(im_,1) * size(im_,2) * 128 ^ p)**  
The factor with which to multiply the gradients of the p-norm.

**'tvNormFactor' :: 1 / (size(im_,1) * size(im_,2) * 128 ^ 2 * 0.01 ^ 2)**  
The factor with which to multiply the gradients of the total variation regularizer.

**'targetFactor' :: 1 / (sum(target(:) .^ 2))**  
The factor with which to multiply the gradients of the l2 loss function.

**'momentum' :: 0.9**  
The value of the momentum. Set zero for no momentum.

**Comments**

Computations are run on the GPU if IM_ is a gpuArray. Otherwise they are run on the CPU.

#### im_ = dagnn_reconstruction(net, im_, target)

Generates an input image which leads to activations equal to target in the last layer of the network. The reconstruction can be controlled by optional regularizers.

**net** The CNN to visualize.
**im_** The image form which to start the reconstruction, e.g. a zero or noise image.
**target** The target of the reconstruction. A multi-dimensional array representing the values of the activation maps of the last layer. The reconstructed image will result in activations which are equal or close to the target.

**dagnn_reconstruction(..., 'opt', value, ...)** takes the following options:

**'InputName' :: Empty**  
Sets the input variable. Only required for DAG networks with more than one input variable.

**'OutputName' :: Empty**  
Sets the output variable. Only required for DAG networks with more than one output variable.

**'Runs' :: 100**  
An integer specifying the number of runs. One forward and the subsequent backpropagation pass form one run.

**'p' :: 6**  
The p value for the p-norm. (Actually it's the p-norm to the power of p.)

**'pNormFactor' :: 1 / (size(im_,1) * size(im_,2) * 128 ^ p)**  
The factor with which to multiply the gradients of the p-norm.

**'tvNormFactor' :: 1 / (size(im_,1) * size(im_,2) * 128 ^ 2 * 0.01 ^ 2)**  
The factor with which to multiply the gradients of the total variation regularizer.

**'targetFactor' :: 1 / (sum(target(:) .^ 2))**  
The factor with which to multiply the gradients of the l2 loss function.

**'momentum' :: 0.9**  
The value of the momentum. Set zero for no momentum.

**Comments**

Computations are run on the GPU if IM_ is a gpuArray. Otherwise they are run on the CPU.

## 3. Citation

[[Content] (#content)] [[Quick Start Guide] (#1-quick-start-guide)] [[Functions] (#2-functions)] [Citation] [[License] (#4-license)]

If you use this library, please cite

````Tex
@inproceedings{gruen16featurevis,
    author    = {Felix Gr{\"{u}}n, Christian Rupprecht, Nassir Navab, Federico Tombari},
    title     = {A Taxonomy and Library for Visualizing Learned Features in Convolutional Neural Networks},
    booktitle = {{ICML} Visualization for Deep Learning Workshop},
    year      = {2016},
    url       = {http://arxiv.org/abs/1606.07757}
}
````

## 4. License

[[Content] (#content)] [[Quick Start Guide] (#1-quick-start-guide)] [[Functions] (#2-functions)] [[Citation] (#3-citation)] [License]

Simplified BSD License

Copyright (c) 2016, Felix Grün
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
