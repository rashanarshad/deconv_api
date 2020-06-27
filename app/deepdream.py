import numpy as np
import sys

from keras.layers import Input, InputLayer, Flatten, Activation, Dense
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D)
from keras.models import Model
from keras.applications import vgg16
import keras.backend as K
K.set_image_data_format('channels_last')

class DInput(object):
    '''
    A class to define forward and backward operation on Input
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Input layer, whose configuration
                   will be used to initiate DInput(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer

    # input and output of Input layer are the same
    def up(self, data):
        '''
        function to operate input in forward pass, the input and output
        are the same
        # Arguments
            data: Data to be operated in forward pass
        # Returns
            data
        '''
        self.up_data = data
        return self.up_data

    def down(self, data):
        '''
        function to operate input in backward pass, the input and output
        are the same
        # Arguments
            data: Data to be operated in backward pass
        # Returns
            data
        '''
        self.down_data = data
        return self.down_data


class DConvolution2D(object):
    '''
    A class to define forward and backward operation on Convolution2D
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Convolution2D layer, whose configuration
                   will be used to initiate DConvolution2D(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer

        weights = layer.get_weights()
        W, b = weights
        config = layer.get_config()

        # Set up_func for DConvolution2D
        input = Input(shape=layer.input_shape[1:])
        output = Convolution2D.from_config(config)(input)
        up_func = Model(input, output)
        up_func.layers[1].set_weights(weights)
        self.up_func = up_func

        # Flip W horizontally and vertically,
        # and set down_func for DConvolution2D
        W = np.transpose(W, (0, 1, 3, 2))
        W = W[::-1, ::-1, :, :]
        config['filters'] = W.shape[3]
        config['kernel_size'] = (W.shape[0], W.shape[1])
        b = np.zeros(config['filters'])
        input = Input(shape=layer.output_shape[1:])
        output = Convolution2D.from_config(config)(input)
        down_func = Model(input, output)
        down_func.layers[1].set_weights((W, b))
        self.down_func = down_func

    def up(self, data):
        '''
        function to compute Convolution output in forward pass
        # Arguments
            data: Data to be operated in forward pass
        # Returns
            Convolved result
        '''
        self.up_data = self.up_func.predict(data)
        return self.up_data

    def down(self, data):
        '''
        function to compute Deconvolution output in backward pass
        # Arguments
            data: Data to be operated in backward pass
        # Returns
            Deconvolved result
        '''
        self.down_data = self.down_func.predict(data)
        return self.down_data


class DPooling(object):
    '''
    A class to define forward and backward operation on Pooling
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Pooling layer, whose configuration
                   will be used to initiate DPooling(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer
        self.poolsize = layer.pool_size

    def up(self, data):
        '''
        function to compute pooling output in forward pass
        # Arguments
            data: Data to be operated in forward pass
        # Returns
            Pooled result
        '''
        [self.up_data, self.switch] = \
            self.__max_pooling_with_switch(data, self.poolsize)
        return self.up_data

    def down(self, data):
        '''
        function to compute unpooling output in backward pass
        # Arguments
            data: Data to be operated in forward pass
        # Returns
            Unpooled result
        '''
        self.down_data = self.__max_unpooling_with_switch(data, self.switch)
        return self.down_data

    def __max_pooling_with_switch(self, input, poolsize):
        '''
        Compute pooling output and switch in forward pass, switch stores
        location of the maximum value in each poolsize * poolsize block
        # Arguments
            input: data to be pooled
            poolsize: size of pooling operation
        # Returns
            Pooled result and Switch
        '''
        switch = np.zeros(input.shape)
        out_shape = list(input.shape)
        row_poolsize = int(poolsize[0])
        col_poolsize = int(poolsize[1])
        out_shape[1] = out_shape[1] // poolsize[0]
        out_shape[2] = out_shape[2] // poolsize[1]
        pooled = np.zeros(out_shape)

        for sample in range(input.shape[0]):
            for dim in range(input.shape[3]):
                for row in range(out_shape[1]):
                    for col in range(out_shape[2]):
                        patch = input[sample,
                                row * row_poolsize: (row + 1) * row_poolsize,
                                col * col_poolsize: (col + 1) * col_poolsize,
                                dim]
                        max_value = patch.max()
                        pooled[sample, row, col, dim] = max_value
                        max_col_index = patch.argmax(axis=-1)
                        max_cols = patch.max(axis=-1)
                        max_row = max_cols.argmax()
                        max_col = max_col_index[max_row]
                        switch[sample,
                               row * row_poolsize + max_row,
                               col * col_poolsize + max_col,
                               dim] = 1
        return [pooled, switch]

    # Compute unpooled output using pooled data and switch
    def __max_unpooling_with_switch(self, input, switch):
        '''
        Compute unpooled output using pooled data and switch
        # Arguments
            input: data to be pooled
            poolsize: size of pooling operation
            switch: switch storing location of each elements
        # Returns
            Unpooled result
        '''
        out_shape = switch.shape
        unpooled = np.zeros(out_shape)
        for sample in range(input.shape[0]):
            for dim in range(input.shape[3]):
                tile = np.ones((switch.shape[1] // input.shape[1],
                                switch.shape[2] // input.shape[2]))
                out = np.kron(input[sample, :, :, dim], tile)
                unpooled[sample, :, :, dim] = out * switch[sample, :, :, dim]
        return unpooled


class DActivation(object):
    '''
    A class to define forward and backward operation on Activation
    '''

    def __init__(self, layer, linear=False):
        '''
        # Arguments
            layer: an instance of Activation layer, whose configuration
                   will be used to initiate DActivation(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer
        self.linear = linear
        self.activation = layer.activation
        input = K.placeholder(shape=layer.output_shape)

        output = self.activation(input)
        # According to the original paper,
        # In forward pass and backward pass, do the same activation(relu)
        self.up_func = K.function(
            [input, K.learning_phase()], [output])
        self.down_func = K.function(
            [input, K.learning_phase()], [output])

    # Compute activation in forward pass
    def up(self, data, learning_phase=0):
        '''
        function to compute activation in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Activation
        '''
        self.up_data = self.up_func([data, learning_phase])[0]
        return self.up_data

    # Compute activation in backward pass
    def down(self, data, learning_phase=0):
        '''
        function to compute activation in backward pass
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Activation
        '''
        self.down_data = self.down_func([data, learning_phase])[0]
        return self.down_data


class DDense(object):
    '''
    A class to define forward and backward operation on Dense
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Dense layer, whose configuration
                   will be used to initiate DDense(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer
        weights = layer.get_weights()
        W, b = weights
        config = layer.get_config()

        # Set up_func for DDense
        input = Input(shape=layer.input_shape[1:])
        output = Dense.from_config(config)(input)
        up_func = Model(input, output)
        up_func.set_weights(weights)
        self.up_func = up_func

        # Transpose W and set down_func for DDense
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        input = Input(shape=self.output_shape[1:])
        output = Dense(units=self.input_shape[1])(input)
        down_func = Model(input, output)
        down_func.set_weights(flipped_weights)
        self.down_func = down_func

    def up(self, data):
        '''
        function to compute dense output in forward pass
        # Arguments
            data: Data to be operated in forward pass
        # Returns
            Result of dense layer
        '''
        self.up_data = self.up_func.predict(data)
        return self.up_data

    def down(self, data):
        '''
        function to compute dense output in backward pass
        # Arguments
            data: Data to be operated in forward pass
        # Returns
            Result of reverse dense layer
        '''
        # data = data - self.bias
        self.down_data = self.down_func.predict(data)
        return self.down_data


class DFlatten(object):
    '''
    A class to define forward and backward operation on Flatten
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Flatten layer, whose configuration
                   will be used to initiate DFlatten(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer
        self.shape = layer.input_shape[1:]
        self.up_func = K.function(
            [layer.input, K.learning_phase()], [layer.output])

    # Flatten 2D input into 1D output
    def up(self, data, learning_phase=0):
        '''
        function to flatten input in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Flattened data
        '''
        self.up_data = self.up_func([data, learning_phase])[0]
        return self.up_data

    # Reshape 1D input into 2D output
    def down(self, data):
        '''
        function to unflatten input in backward pass
        # Arguments
            data: Data to be operated in backward pass
        # Returns
            Recovered data
        '''
        new_shape = [data.shape[0]] + list(self.shape)
        assert np.prod(self.shape) == np.prod(data.shape[1:])
        self.down_data = np.reshape(data, new_shape)
        return self.down_data


def find_top_filters(output, top=8):
    filter_sum = []
    for filter_index in range(output.shape[-1]):
        if output.ndim == 2:
            sum_value = np.sum(output[:, filter_index])
        else:
            sum_value = np.sum(output[:, :, :, filter_index])
        if sum_value > 0:
            filter_sum.append((filter_index, sum_value))
    filter_sum.sort(key=lambda x: x[1], reverse=True)
    # print(len(filter_sum))
    return filter_sum[:top]


def visualize_all_layers(model, data, layer_name='predictions', visualize_mode='all'):
    '''
    function to visualize feature
    # Arguments
        model: Pre-trained model used to visualize data
        data: image to visualize
        layer_name: Name of layer to visualize
        feature_to_visualize: Features to visualize
        visualize_mode: Visualize mode, 'all' or 'max', 'max' will only pick
                        the greates activation in a feature map and set others
                        to 0s, this will indicate which part fire the neuron
                        most; 'all' will use all values in a feature map,
                        which will show what image the filter sees. For
                        convolutional layers, There is difference between
                        'all' and 'max', for Dense layer, they are the same
    # Returns
        The image reflecting feature
    '''
    deconv_layers = []
    # Stack layers
    for layer in model.layers:
        if isinstance(layer, Convolution2D):
            deconv_layers.append((layer.name, DConvolution2D(layer)))
            deconv_layers.append((layer.name + '_activation', DActivation(layer)))
        elif isinstance(layer, MaxPooling2D):
            deconv_layers.append((layer.name, DPooling(layer)))
        elif isinstance(layer, Dense):
            deconv_layers.append((layer.name, DDense(layer)))
            deconv_layers.append((layer.name + '_activation', DActivation(layer)))
        elif isinstance(layer, Activation):
            deconv_layers.append((layer.name, DActivation(layer)))
        elif isinstance(layer, Flatten):
            deconv_layers.append((layer.name, DFlatten(layer)))
        elif isinstance(layer, InputLayer):
            deconv_layers.append((layer.name, DInput(layer)))
        else:
            print('Cannot handle this type of layer')
            print(layer.get_config())
            sys.exit()
        if layer_name == layer.name:
            break

    # Forward pass
    deconv_layers[0][1].up(data)
    for i in range(1, len(deconv_layers)):
        deconv_layers[i][1].up(deconv_layers[i - 1][1].up_data)

    # Selecting layers to visualize
    layers_to_visualize = []
    model_layers = set([layer.name for layer in model.layers])
    layers_to_visualize = [x for x, y in enumerate(deconv_layers)
                           if y[0] in model_layers]
    layers_to_visualize.reverse()
    # Removing the input layer
    layers_to_visualize.pop()
    print('layers_to_visualize:', layers_to_visualize)

    deconv_dict = dict()
    for i in layers_to_visualize:
        deconv_list = []
        output = deconv_layers[i][1].up_data
        top_filters = find_top_filters(output)
        print('output.shape :', output.shape)
        print('deconv_layer:', deconv_layers[i][0])
        print('top_filters:', top_filters)
        for feature_to_visualize, sum_value in top_filters:
            assert output.ndim == 2 or output.ndim == 4
            if output.ndim == 2:
                feature_map = output[:, feature_to_visualize]
            else:
                feature_map = output[:, :, :, feature_to_visualize]
            if 'max' == visualize_mode:
                max_activation = feature_map.max()
                temp = feature_map == max_activation
                feature_map = feature_map * temp
            elif 'all' != visualize_mode:
                print('Illegal visualize mode')
                sys.exit()
            output_temp = np.zeros_like(output)
            if 2 == output.ndim:
                output_temp[:, feature_to_visualize] = feature_map
            else:
                output_temp[:, :, :, feature_to_visualize] = feature_map

            # Backward pass
            deconv_layers[i][1].down(output_temp)
            for j in range(i - 1, -1, -1):
                deconv_layers[j][1].down(deconv_layers[j + 1][1].down_data)
            deconv = deconv_layers[0][1].down_data
            deconv = deconv.squeeze()
            deconv_list.append(deconv)
        deconv_dict[deconv_layers[i][0]] = deconv_list

    return deconv_dict






def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

