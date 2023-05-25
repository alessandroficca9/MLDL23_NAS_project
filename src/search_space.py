from torch import nn




class ConvolutionalBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):  # aggiungere kernel, stride, padding, groups
        super(ConvolutionalBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
   
    
class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels) 
            )
        
        self.relu = nn.ReLU(inplace=True)

         # Utilizzato per aggiungere la shortcut connection solo se i canali di input e output sono diversi
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
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
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
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


## ConvNeXt Block 
class ConvNeXt(nn.Module):

    def __init__(self, inp_dim):
        super(ConvNeXt, self).__init__()

        self.dwConv = nn.Sequential(
            # depthwise conv 7x7
            nn.Conv2d(inp_dim, inp_dim, kernel_size=7, padding=3, groups=inp_dim, bias=False),
            nn.BatchNorm2d(inp_dim, eps=1e-6),
        )        

        self.pwConv = nn.Sequential(
            nn.Conv2d(inp_dim, 4*inp_dim, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )

        self.pwConv2 = nn.Conv2d(4*inp_dim, inp_dim,kernel_size=1, stride=1)

    def forward(self, x):
        input = x

        x = self.dwConv(x)
        x = self.pwConv(x)
        x = self.pwConv2(x)

        x += input
        return x 



