from search_space import *
import random


building_blocks = {
    "ConvBNReLU" : ConvBNReLU,
    "InvertedResidual" : InvertedResidual,
    "DWConv" : DWSConv,
    "BottleneckResidual" : BottleneckResidual,
    "ConvNeXt" : ConvNeXt,
    "AvgPool2x2" : nn.AvgPool2d(kernel_size=2),
    "MaxPool2x2" : nn.MaxPool2d(kernel_size=2)
}

channels = [16, 32, 64, 96, 160, 320]

def random_search(num_iterations, inp_channel_first):

    blocks = []
    inp_ch = inp_channel_first

    for i in range(num_iterations):
        num_blocks = random.randint(1,9)

        for j in num_blocks:
            block = random.choice(building_blocks.keys)
            out_ch = random.choice(channels)

            blocks.append( (block, inp_ch, out_ch))
            inp_ch = out_ch


#def make_architecture(blocks):



