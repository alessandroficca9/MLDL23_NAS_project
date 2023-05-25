from search_space import *
import random
import math 




building_blocks = {
    #"ConvBNReLU" : ConvolutionalBlock,
    "InvertedResidual" : InvertedResidualBlock,
    "DWConv" : DepthwiseSeparableConvBlock,
    "BottleneckResidual" : BottleneckResidualBlock,
    "ConvNeXt" : ConvNeXt,
    #"AdaptiveAvgPool2x2" : nn.AdaptiveAvgPool2d,
    #"AdaptiveMaxPool2x2" : nn.MaxPool2d,
}

channels = [16, 32, 64, 96, 160, 320]

class NetworkDecoded(nn.Module):

    def __init__(self, network_encoding, num_classes):
        super(NetworkDecoded, self).__init__()

        self.layers = nn.Sequential()

        for block_type, input_ch, output_ch in network_encoding:
            if block_type == "ConvNeXt":
                self.layers.append(
                    building_blocks[block_type](input_ch)
                )
            else:
                self.layers.append(
                    building_blocks[block_type](input_ch,output_ch) )
            

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(output_ch, num_classes) )
        
        # convert to half precision
        #self.half()

        # convert BatchNorm to float32
        #for layer in self.modules():
        #    if isinstance(layer, nn.BatchNorm2d):
        #        layer.float()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            _, _, h, w = x.size()
            x = nn.AdaptiveAvgPool2d((h, w))(x)  # Adattamento delle dimensioni spaziali

        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)     # forse togliere adaptive avg pool finale??
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
      
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
              m.weight.data.normal_(0, math.sqrt(2. / n))
              if m.bias is not None:
                    m.bias.data.zero_()
          elif isinstance(m, nn.BatchNorm2d):
              m.weight.data.fill_(1)
              m.bias.data.zero_()
          elif isinstance(m, nn.Linear):
              m.weight.data.normal_(0, 0.01)
              m.bias.data.zero_()




def generate_random_network(input_channels_first, num_max_blocks):

    blocks = []
    input_channels = input_channels_first
    
    for _ in range(random.randint(1,num_max_blocks)):
        block_type = random.choice(list(building_blocks.keys()))
        if block_type == "ConvNeXt":
            output_channels = input_channels
        else:
            output_channels = random.choice(channels)
        blocks.append( (block_type, input_channels, output_channels))
        input_channels = output_channels
        
    return blocks


def random_search(num_iterations, num_max_blocks):

    networks_encoded = [ generate_random_network(input_channels_first=3, num_max_blocks=num_max_blocks) for _ in range(num_iterations)]
    architectures = [ NetworkDecoded(enc, num_classes=2) for enc in networks_encoded]

    return architectures




