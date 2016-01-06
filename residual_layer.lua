require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'

function addDiff_ResidualLayer(input, nChannels, nOutChannels, ctype, stride, expand)
  local net
  if expand then
    net = ctype(nChannels, nOutChannels, 4, 4, 2, 2, 1, 1)(input)
  else
    net = ctype(nChannels, nOutChannels, 4, 4)(input)
  end
  net = nn.SpatialBatchNormalization(nOutChannels)(net)
  net = cudnn.ReLU(true)(net)
  net = ctype(nOutChannels, nOutChannels, 3,3, 1, 1, 1, 1)(net)
  net = nn.SpatialBatchNormalization(nOutChannels)(net)

  local skip = input
  if expand then
    skip = ctype(nChannels, nOutChannels, 4, 4, 2, 2, 1, 1)(skip)
  else
    skip = ctype(nChannels, nOutChannels, 4, 4)(skip)
  end  
  skip = nn.SpatialBatchNormalization(nOutChannels)(skip)

  net = nn.CAddTable(){net, skip}
  net = cudnn.ReLU(true)(net)
  return skip
end


--src: https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua
function addConvResidualLayer(input,  nChannels, nOutChannels, stride, ctype)
   --[[
   Residual layers! Implements option (A) from Section 3.3. The input
   is passed through two 3x3 convolution filters. In parallel, if the
   number of input and output channels differ or if the stride is not
   1, then the input is downsampled or zero-padded to have the correct
   size and number of channels. Finally, the two versions of the input
   are added together.
               Input
                 |
         ,-------+-----.
   Downsampling      3x3 convolution+dimensionality reduction
        |               |
        v               v
   Zero-padding      3x3 convolution
        |               |
        `-----( Add )---'
                 |
              Output
   --]]
   -- Path 1: Convolution
   -- The first layer does the downsampling and the striding
   local net = ctype(nChannels, nOutChannels,
                                           3,3, stride, stride, 1, 1)(input)
   net = nn.SpatialBatchNormalization(nOutChannels)(net)
   net = cudnn.ReLU(true)(net)
   net = ctype(nOutChannels, nOutChannels,
                                      3,3, 1, 1, 1, 1)(net)
   net = nn.SpatialBatchNormalization(nOutChannels)(net)

   -- Should we put Batch Normalization here? I think not, because
   -- BN would force the output to have unit variance, which breaks the residual
   -- property of the network.
   -- What about ReLU here? I think maybe not for the same reason. Figure 2
   -- implies that they don't use it here

   -- Path 2: Identity / skip connection
   local skip = input
   if stride > 1 then
       -- optional downsampling
       skip = nn.SpatialAveragePooling(1, 1, stride, stride)(skip)
   end
   if nOutChannels > nChannels then
       -- optional padding
       skip = nn.Padding(1, (nOutChannels - nChannels), 3)(skip)
   end

   -- Add them together
   net = nn.CAddTable(){net, skip}
   net = cudnn.ReLU(true)(net) --TODO
   -- ^ don't put a ReLU here! see http://gitxiv.com/comments/7rffyqcPLirEEsmpX

   return net
end
