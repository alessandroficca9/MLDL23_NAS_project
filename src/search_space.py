from torch import nn


class ConvolutionalBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):  # aggiungere kernel, stride, padding, groups
        super(ConvolutionalBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
   
    
class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU6(inplace=True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU6(inplace=True)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels) 
            )
        
        self.relu = nn.ReLU6(inplace=True)

         # Utilizzato per aggiungere la shortcut connection solo se i canali di input e output sono diversi
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        x = self.conv1x1_1(x)
        x = self.conv3x3(x)
        x = self.conv1x1_2(x)
        
        x += self.shortcut(identity)
        x = self.relu(x)
        return x
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(in_channels * expansion_factor)
        self.use_res_connect = in_channels == out_channels
        

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        identity = x
        
        x = self.conv1x1_1(x)
        x = self.depthwise(x)
        x = self.conv1x1_2(x)
        
        if self.use_res_connect:
            x += identity
        x = self.relu(x)

        return x


"""
## ConvNeXt Block 
class ConvNeXt(nn.Module):

    def __init__(self, inp_dim):
        super(ConvNeXt, self).__init__()

        self.dwConv = nn.Sequential(
            # depthwise conv 7x7
            nn.Conv2d(inp_dim, inp_dim, kernel_size=7, padding=3, groups=inp_dim),
            nn.LayerNorm(inp_dim, eps=1e-6),
        )        

        self.pwConv = nn.Sequential(
            nn.Conv2d(inp_dim, 4*inp_dim, kernel_size=1, stride=1),
            nn.GELU()
        )

        self.pwConv2 = nn.Conv2d(4*inp_dim, inp_dim,kernel_size=1, stride=1)

    def forward(self, x):
        input = x

        x = self.dwConv(x)
        x = self.pwConv(x)
        x = self.pwConv2(x)

        x += input
        return x 


"""


"""
## "Classical" convolutional block
class ConvBNReLU(nn.Sequential):

  def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,padding=0, groups=1):  

    
    groups:  controls the connections between inputs and outputs
    if groups = 1 classic convolution
    if groups = inp_ch  => depthwise conv 
    
    padding = (kernel_size-1)//2

    super(ConvBNReLU,self).__init__(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride,padding, groups=groups,bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True)
    )
    
## Inverted Residual Block (MobileNetV2)
class InvertedResidual(nn.Module):

    def __init__(self, inp_ch, out_ch, expand_ratio=6, stride=1):
    
        
        input_ch: number of channel in input
        output_ch: number of channel in output
        exp_ratio (t): expanand the input channel t time (expansion)
        
        super(InvertedResidual,self).__init__()

        self.stride = stride
        hidden_dim = int(round(expand_ratio*inp_ch))

        self.use_res_connect = self.stride == 1 and inp_ch == out_ch 

        layers = []

        if expand_ratio != 1:
            # pointwise conv ReLU -> expansion (t != 1)
            layers.append(ConvBNReLU(inp_ch, hidden_dim, kernel_size=1, stride=1, groups=1))
        
        layers.extend([
            # depthwise conv ReLU 3x3
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride,padding=1, groups=hidden_dim),
            # pointwise linear conv (projection)
            nn.Conv2d(hidden_dim, out_ch, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
## Depthwise separable convolutional block
class DWSConv(nn.Module):

    def __init__(self, inp_ch, out_ch, kernel_size=3, stride=1, padding=0):
        super(DWSConv,self).__init__()
        
        
        
        self.dwConv = nn.Sequential(
            nn.Conv2d(inp_ch, inp_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp_ch),
            nn.BatchNorm2d(inp_ch),
            nn.ReLU6(inplace=True)
        )
        self.pwConv = nn.Sequential(
            nn.Conv2d(inp_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        x = self.dwConv(x)
        x = self.pwConv(x)
        return x
        

class BottleneckResidual(nn.Module):

    def __init__(self, inp_ch, out_ch, expansion_factor=4):

        super(BottleneckResidual, self).__init__()

        hidden_dim = out_ch // expansion_factor

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp_ch, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
        )

        self.relu = nn.ReLU6(inplace=True)

        # Utilizzato per aggiungere la shortcut connection solo se i canali di input e output sono diversi
        if inp_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += self.shortcut(identity)
        x = self.relu(x)
        return x
    

## ConvNeXt Block 
class ConvNeXt(nn.Module):

    def __init__(self, inp_dim):
        super(ConvNeXt, self).__init__()

        self.dwConv = nn.Sequential(
            # depthwise conv 7x7
            nn.Conv2d(inp_dim, inp_dim, kernel_size=7, padding=3, groups=inp_dim),
            nn.LayerNorm(inp_dim, eps=1e-6),
        )        

        self.pwConv = nn.Sequential(
            nn.Conv2d(inp_dim, 4*inp_dim, kernel_size=1, stride=1),
            nn.GELU()
        )

        self.pwConv2 = nn.Conv2d(4*inp_dim, inp_dim,kernel_size=1, stride=1)

    def forward(self, x):
        input = x

        x = self.dwConv(x)
        x = self.pwConv(x)
        x = self.pwConv2(x)

        x += input
        return x 


"""

