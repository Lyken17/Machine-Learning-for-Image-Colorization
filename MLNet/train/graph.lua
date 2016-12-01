require 'nn'
require 'nngraph'
require 'image'

local infile  = arg[1]
local outfile = arg[2] or 'out.png'

local d        = torch.load( 'colornet.t7' )
local datamean = d.mean
local model    = d.model:float()

graph.dot(model.fg, 'Forward Graph', 'fg')
