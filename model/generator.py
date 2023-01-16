import functools
import torch
import torch.nn as nn
from non_local_embedded_gaussian import NONLocalBlock2D

class Generator(nn.Module):
    """
    The input only contains Pressure or temperature feature
    base = 'P': input Pressure features
    base = 'T': input Temperature features
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', base='P'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            base                -- input Pressure or Temperature features: 'P' or 'T'
        """
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Add downsampling layer
        downsample_layer = [nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model1 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model1 += [nn.ReflectionPad2d(3)]
        model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model1 += [nn.Tanh()]

        self.base = base
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        self.downsample_layer = nn.Sequential(*downsample_layer)

    def forward(self, input):
        """Standard forward"""
        if self.base == 'T':
            input = self.downsample_layer(input)
        encoder = self.model(input)
        decoder = self.model1(encoder)
        return decoder


class PTGenerator(nn.Module):
    """ input pressure and temperature features, concatenate them together through channels in the beginning
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(PTGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Add downsampling layer
        downsample_layer = [nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)]

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc + input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model1 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model1 += [nn.ReflectionPad2d(3)]
        model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model1 += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.downsample_layer = nn.Sequential(*downsample_layer)
        self.model1 = nn.Sequential(*model1)

    def forward(self, input, input2):
        """Standard forward"""
        down = self.downsample_layer(input2)
        # concatenate pressure and temperature together through channels
        cat_layer = torch.cat((input, down), 1)
        encoder = self.model(cat_layer)
        decoder = self.model1(encoder)
        return decoder


class PTLatentGenerator(nn.Module):
    """ input pressure and temperature features, concatenate them together through channels in the Latent space
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(PTLatentGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Add downsampling layer
        downsample_layer = [nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)]

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # deduce channel number
        reduce_layer = [nn.Conv2d(2 * ngf * mult, ngf * mult, kernel_size=1, stride=1, padding=0)]

        model1 = []

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model1 += [nn.ReflectionPad2d(3)]
        model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model1 += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.downsample_layer = nn.Sequential(*downsample_layer)
        self.reduce_layer = nn.Sequential(*reduce_layer)
        self.model1 = nn.Sequential(*model1)

    def forward(self, input, input2):
        """Standard forward"""
        down = self.downsample_layer(input2)
        encoder = self.model(input)
        encoder2 = self.model(down)
        # concatenate pressure and temperature together through channels
        cat_layer = torch.cat((encoder, encoder2), 1)
        reduce_encoder = self.reduce_layer(cat_layer)
        decoder = self.model1(reduce_encoder)
        return decoder


class AttentionGenerator(nn.Module):
    """Encoder1 based generator that consists of Attention layer between downsampling/upsampling operation.
    base = 'P': Encoder1: pressure features
                Encoder2: temperature features
                Attention Layer: merge temperature features into pressure features
    base = 'T': Encoder1: temperature features
                Encoder2: pressure features
                Attention Layer: merge pressure features into temperature features
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', base='P'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            base                -- Pressure based or Temperature based Attention model: 'P' or 'T'
        """
        assert(n_blocks >= 0)
        super(AttentionGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Add downsampling layer
        downsample_layer = [nn.Conv2d(input_nc, input_nc, kernel_size=2, stride=2, padding=0)]

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # Add attention non_local layer
        non_local = NONLocalBlock2D(ngf * mult, sub_sample=True, bn_layer=True)

        model1 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model1 += [nn.ReflectionPad2d(3)]
        model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model1 += [nn.Tanh()]

        self.base = base
        self.model = nn.Sequential(*model)
        self.non_local = non_local
        self.model1 = nn.Sequential(*model1)
        self.downsample_layer = nn.Sequential(*downsample_layer)

    def forward(self, input, input2):
        """Standard forward"""
        down = self.downsample_layer(input2)
        encoder = self.model(input)
        encoder2 = self.model(down)
        if self.base == 'T':
            encoder, encoder2 = encoder2, encoder
        attention_layer = self.non_local(encoder, encoder2)
        decoder = self.model1(attention_layer)
        return decoder


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out