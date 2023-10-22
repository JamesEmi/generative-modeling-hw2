import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=1,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        #Changed padding to zero - will it work??
        self.upscale_factor = upscale_factor

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel-wise upscale_factor^2 times
        # 2. Use torch.nn.PixelShuffle to form an output of dimension
        # (batch, channel, height*upscale_factor, width*upscale_factor)
        # 3. Apply convolution and return output
        ##################################################################
        # 1. Repeat x channel-wise upscale_factor^2 times
        x = x.repeat_interleave(int(self.upscale_factor ** 2), dim=1)
        
        # 2. Use torch.nn.PixelShuffle
        x = F.pixel_shuffle(x, self.upscale_factor)
        
        # 3. Apply 2D Conv to upsampled image and return output
        return self.conv(x)     
        
        
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=1, stride=1
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding, stride=stride
        )
        self.downscale_ratio = downscale_ratio

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use torch.nn.PixelUnshuffle to form an output of dimension
        # (batch, channel, downscale_factor^2, height, width)
        # 2. Then split channel-wise into
        # (downscale_factor^2, batch, channel, height, width) images
        # 3. Take the average across dimension 0, apply convolution,
        # and return the output
        ##################################################################
        
        #1
        x = F.pixel_unshuffle(x, self.downscale_ratio)
        #check if this ok or if I have to permute with something like x = x = x.permute(0, 1, 3, 5, 2, 4)
        
        #2
        # x = x.reshape(x.shape[0], -1, int(self.downscale_ratio ** 2), x.shape[2], x.shape[3])
        x = x.view(x.size(0), x.size(1) // int(self.downscale_ratio**2), int(self.downscale_ratio**2), x.size(2), x.size(3))
        
        #3
        # x = x.mean(dim=1, keepdim=True)
        x = x.mean(2)
        return self.conv(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockUp(torch.jit.ScriptModule):
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128, stride=1):
        super(ResBlockUp, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels), #probably need all the args here that the comments have.
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=1, bias=False), #careful with the padding here.
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            UpSampleConv2D(n_filters, kernel_size=kernel_size, n_filters=n_filters)) # Refer earlier definition of UpSampleConv2D
            #Refer comments above and update the code. layers ok??
        
        self.upsample_residual = UpSampleConv2D(input_channels, kernel_size=3, n_filters=n_filters) #Changing kernel size from 1 to 3.
        
        #HAVEN't USED THE PARAMS GIVEN FOR BATCHNORM. CHECK if okay, or if to add them. 
        
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to upsample the residual before adding it
        # to the layer output.
        ##################################################################
        # out = self.layers(x)
        # residual = self.upsample_residual(x)
        return self.layers(x).add_(self.upsample_residual(x))
#         print("Before layers:", x.shape)
#         x1 = self.layers(x)
#         print("After layers:", x1.shape)
#         x2 = self.upsample_residual(x)
#         print("After upsample_residual:", x2.shape)
        
#         return self.layers(x) + self.upsample_residual(x)
    
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockDown(torch.jit.ScriptModule):
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=1, padding=1), #check kernel_size, should it be 3, or (3,3)??
            nn.ReLU(),
            DownSampleConv2D(input_channels = n_filters, n_filters = n_filters, kernel_size=kernel_size)
        ) #Try and streamline that last part; had to do it explicitly because the order was all messed up.
            
        self.downsample_residual = DownSampleConv2D(input_channels, n_filters = n_filters, kernel_size=3, stride=1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to downsample the residual before adding
        # it to the layer output.
        ##################################################################
        # out = self.layers(x)
        # residual = self.downsample_residual(x)
        
        # return self.layers(x).add_(self.downsample_residual(x))
        return self.layers(x) + self.downsample_residual(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlock(torch.jit.ScriptModule):
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, n_filters=128, kernel_size=3):
        super(ResBlock, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the conv layers. Don't forget the residual
        # connection!
        ##################################################################
        return x.add_(self.layers(x))
        # add inplace operation here and see if it works.
        # For this addtion to work, num of input channels must be the same as num of output channels
        # i.e, n_filters must be same as input_channels!
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class Generator(torch.jit.ScriptModule):
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        
        # Reshape 2048 to (512, 4, 4)
        self.dense = nn.Linear(128, 2048, bias=True) #is the bias=True part needed?
        # Check if it has to be Linear(in_features=128, out_features=2048, bias=True)
        
        self.layers = nn.Sequential(
            ResBlockUp(input_channels=128, n_filters=128),
            ResBlockUp(input_channels=128, n_filters=128),
            ResBlockUp(input_channels=128, n_filters=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),  # Output 3 channels for RGB
            nn.Tanh()
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward_given_samples(self, z):
        ##################################################################
        # TODO 1.1: Forward the generator assuming a set of samples z has
        # been passed in. Don't forget to re-shape the output of the dense
        # layer into an image with the appropriate size!
        ##################################################################
        x = self.dense(z)
        x = x.view(x.size(0), 128, 4, 4) # Reshape to image
        return self.layers(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, z, n_samples: int = 1024):
        ##################################################################
        # TODO 1.1: Generate n_samples latents and forward through the
        # network.
        ##################################################################
        z = torch.randn((n_samples, 128)).cuda()  # Assuming the latent space is 128-dimensional
        return self.forward_given_samples(z)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class Discriminator(torch.jit.ScriptModule):
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        
        self.layers = nn.Sequential(
            ResBlockDown(input_channels=3, n_filters=128),
            ResBlockDown(input_channels=128, n_filters=128),
            ResBlock(input_channels=128, n_filters=128),
            ResBlock(input_channels=128, n_filters=128),
            nn.ReLU()
        )
        
        # A final dense layer to produce a single output for real/fake classification
        self.dense = nn.Linear(128, 1)
        #or is it nn.Linear(8192,1)? 
        
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the discriminator assuming a batch of images
        # have been passed in. Make sure to sum across the image
        # dimensions after passing x through self.layers.
        ##################################################################
        
        x = self.layers(x)
        # Flatten the feature map while preserving the batch size
        x = x.view(x.size(0), -1)
        # Only use the depth of 128 for each image in the batch
        x = x.mean(dim=1)
        
        #pass through dense layer
        x = self.dense(x)
        
        return x        
        
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
