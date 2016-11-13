require 'nn'
require 'nngraph'
require 'image'

local cmd = torch.CmdLine()

cmd:option('-input',   'example.png',     'input gray scale image')
cmd:option('-output',  'out.png',         'output gray scale image')
cmd:option('-GPU',            0,         'Use GPUs or not, default not')

local opt = cmd:parse(arg or {})

local infile  = opt.input
local outfile = opt.output

local d        = torch.load( 'colornet.t7' )
local datamean = d.mean
local model    = d.model:float()

local function pred2rgb( x, data )
   local I = torch.cat( data[1][{{1},{},{}}]:float(),
                        data[1]:clone():float():mul(2):add(-1), 1)
   local O = image.scale( I, x:size(3), x:size(2) )
   local X = image.rgb2lab( image.yuv2rgb( torch.repeatTensor( x, 3, 1, 1 ) ) )
   O    = O*100
   O[1] = X[1]
   O    = image.rgb2yuv( image.lab2rgb( O ) )
   return image.yuv2rgb( torch.cat( x, O[{{2,3},{},{}}], 1 ) )
end


local I = image.load( infile )

if I:size(1)==3 then I = image.rgb2y(I) end

local X2 = image.scale( I, torch.round(I:size(3)/8)*8, torch.round(I:size(2)/8)*8 ):add(-datamean):float()
local X1 = image.scale( X2, 224, 224 ):float()

X1 = X1:reshape( 1, X1:size(1), X1:size(2), X1:size(3) )
X2 = X2:reshape( 1, X2:size(1), X2:size(2), X2:size(3) )


model.forwardnodes[9].data.module.modules[3].nfeatures = X2:size(3)/8
model.forwardnodes[9].data.module.modules[4].nfeatures = X2:size(4)/8

image.save( outfile, pred2rgb( I:float(), model:forward( {X1, X2} ) ) )
