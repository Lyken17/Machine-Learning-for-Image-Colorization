local NetTools = require 'model'


net = NetTools:build_model(1)

img1 = torch.rand(1, 1, 256, 256)
img2 = torch.rand(1, 1, 224, 224)


net:forward({img1, img2})
graph.dot(net.fg, 'temp', 'temp')
