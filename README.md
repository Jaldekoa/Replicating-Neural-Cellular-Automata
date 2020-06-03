# Replicating Neural Cellular Automata
Original work: https://distill.pub/2020/growing-ca/

This neural network learn to recreate the lizard emoji image. It's size is 64x64 pixels in the RGBA format. The main goal of this project is to learn more about Deep Learning and Tensorflow. This model is not original, only is my attempt to replicate an existing model.

## The Model
![Model Neural Cellular Automata](https://github.com/Jaldekoa/Replicating-Neural-Cellular-Automata/blob/master/Img/Model.PNG)

The input image has the shape of (64, 64, 4) with 3 channels for the RGB color and the forth is the alpha channel. First, I expand the third dimension to 16. Then, I create an edge detector filter (sobelX and sobelY) and a cell identity filter. Next, I send the increased image to a non trainable Depthwise Conv2D network (tf.nn.depthwise_conv2d) with strides = 1 and same padding. In this way, I have the perception vector for every single cell (or pixel). Now, I send the perception tensor through a Conv2D network with a relu activation and next through another Conv2D without activation. These are equivalent for a Dense network in any pixel. This provides an updating mask which will added to the input image following an stochastic variable. Finally, an "alive mask" does the cellular automata work for alpha cannel more than one. The loss is calculated with the mean squared error function.

## The Result

The result obtained in 5,000 epochs with only one batch is the following.
[Input and Expected Output](https://github.com/Jaldekoa/Replicating-Neural-Cellular-Automata/blob/master/Img/Input-Output.jpg)
<p align="center">
  <img src="https://github.com/Jaldekoa/Replicating-Neural-Cellular-Automata/blob/master/Img/Replicating%20Neural%20Cellular%20Automata.gif"alt="Training the model"/>
</p>
