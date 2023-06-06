from torch.nn.modules.batchnorm import BatchNorm2d
from torch import nn  
import math 

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):

  def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):   # add norm layer?

    """
    groups:  controls the connections between inputs and outputs
    if groups = 1 classic convolution
    if groups = inp_ch  => depthwise conv 
    """
    padding = (kernel_size-1)//2
    super(ConvBNReLU,self).__init__(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups,bias=False),
        BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True)
    )
        

class InvertedResidual(nn.Module):

  def __init__(self, input_ch, output_ch, stride, expand_ratio):
    
    """
    input_ch: number of channel in input
    output_ch: number of channel in output
    exp_ratio (t): expanand the input channel t time (expansion)
    """
    super(InvertedResidual,self).__init__() 
    self.stride = stride
    hidden_dim = int(round(expand_ratio*input_ch))

    self.use_res_connect = self.stride == 1 and input_ch == output_ch 

    layers = []

    if expand_ratio != 1:
      # pointwise conv ReLU -> expansion (t != 1)
      layers.append(ConvBNReLU(input_ch, hidden_dim, kernel_size=1, stride=1, groups=1))
    
    layers.extend([
        # depthwise conv ReLU
        ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
        # pointwise linear conv
        nn.Conv2d(hidden_dim, output_ch, kernel_size=1,stride=1,padding=0,bias=False),
        BatchNorm2d(output_ch)
    ])

    self.conv = nn.Sequential(*layers)

  def forward(self, x):
    if self.use_res_connect:
      return x + self.conv(x)
    else:
      return self.conv(x)


class MobileNetV2(nn.Module):

  def __init__(self, num_classes=2, width_mult=0.35, round_nearest=8):    # add norm layer? block?

    super(MobileNetV2,self).__init__()
    
    #hyperparams
    input_channel = 32
    last_channel = 1280

    inverted_residual_setting = [
        # t, c, n, s
        [1,16,1,1],
        [6,24,2,2],
        [6,32,3,2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    # building the first layer
    input_channel = _make_divisible(input_channel * width_mult, round_nearest)
    self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest) 
    features = [ConvBNReLU(3, input_channel, stride=2)]

    # building Bottleneck residual block
    for t,c,n,s in inverted_residual_setting:
      output_channel = _make_divisible(c*width_mult, round_nearest)
      for i in range(n):
        stride = s if i==0 else 1
        features.append(InvertedResidual(input_channel, output_channel,stride=stride,expand_ratio=t))
        input_channel = output_channel 

    # building last layers
    features.append( ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
    # make it sequential
    self.features = nn.Sequential(*features)

    # building classifier
    self.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(self.last_channel, num_classes)
    )

    #weight initialization
    self._initialize_weights()

    
    # convert to half precision
    #self.half()

    # convert BatchNorm to float32
    # for layer in self.modules():
    #   if isinstance(layer, nn.BatchNorm2d):
    #     layer.float()

  def _forward_impl(self, x):
        
    x = self.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
    x = self.classifier(x)
    return x

  def forward(self, x):
      return self._forward_impl(x)
  
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

