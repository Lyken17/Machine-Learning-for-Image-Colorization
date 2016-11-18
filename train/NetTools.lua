require 'nn'
require 'nngraph'
require 'image'

local M = { }

function M:build(opts)
  -- Convolution = cudnn.SpatialConvolution
  Convolution = nn.SpatialConvolution

  -- Avg = cudnn.SpatialAveragePooling
  Avg = nn.SpatialAveragePooling

  -- ReLU = cudnn.ReLU
  ReLU = nn.ReLU

  Max = nn.SpatialMaxPooling

  SBatchNorm = nn.SpatialBatchNormalization


  local function vgg()
      model = nn.Sequential()
      model:add(Convolution(  1,  64, 3,3, 2,2, 1,1))
      model:add(ReLU(true))
      model:add(Convolution( 64, 128, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(Convolution(128, 128, 3,3, 2,2, 1,1))
      model:add(ReLU(true))
      model:add(Convolution(128, 256, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(Convolution(256, 256, 3,3, 2,2, 1,1))
      model:add(ReLU(true))
      model:add(Convolution(256, 512, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      return model
  end

  local function global_feature()
      --[[

      ]]--
      model = nn.Sequential()
      model:add(Convolution(512, 512, 3,3, 2,2, 1,1))
      model:add(ReLU(true))
      model:add(Convolution(512, 512, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(Convolution(512, 512, 3,3, 2,2, 1,1))
      model:add(ReLU(true))
      model:add(Convolution(512, 512, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(nn.View(-1, 25088))
      model:add(nn.Linear(25088, 1024))
      model:add(ReLU(true))
      model:add(nn.Linear(1024, 512))
      model:add(ReLU(true))
      return model
  end

  local function classification(nLabels)
      model = nn.Sequential()
      model:add(nn.Linear(512, 512))
      model:add(ReLU(true))
      model:add(nn.Dropout(0.5))

      model:add(nn.Linear(512, nLabels))
      model:add(nn.LogSoftMax())
      return model
  end


  local function mid_level_feature()
      model = nn.Sequential()
      model:add(Convolution(512, 512, 3, 3, 1, 1, 1, 1))
      model:add(ReLU(true))
      model:add(Convolution(512, 256, 3, 3, 1, 1, 1, 1))
      model:add(ReLU(true))
      return model
  end

  local function img2feat(h, w)
      h_feat = h / 8
      w_feat = w / 8

      model = nn.Sequential()
      model:add(nn.Linear(512, 256))
      model:add(ReLU(true))
      model:add(nn.Replicate(h_feat, 2, 1))
      model:add(nn.Replicate(w_feat, 2, 2))
      return model
  end

  local function upsample_and_color()
      model = nn.Sequential()

      model:add(Convolution(512, 256, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(Convolution(256, 128, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(nn.SpatialUpSamplingNearest(2))

      model:add(Convolution(128, 64, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(Convolution( 64, 64, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(nn.SpatialUpSamplingNearest(2))

      model:add(Convolution( 64, 32, 3,3, 1,1, 1,1))
      model:add(ReLU(true))
      model:add(Convolution( 32,  2, 3,3, 1,1, 1,1))
      model:add(nn.Sigmoid())
      return model
  end

  local function construct(h, w)

    -- two inputs
    -- local origin_img = torch.rand(nSamples,1,256,256)
    -- local scaled_img = torch.rand(nSamples,1,224,224)

    -- local feature / original image
    local low_feat1 = vgg()()
    local mid_feat = mid_level_feature()(low_feat1)

    -- global feature / scaled image
    local low_feat2 = vgg()()
    local global_feat = global_feature()(low_feat2)
    local fusion_layer = img2feat(h, w)(global_feat)

    -- classification
    local predict_label = classification(10)(global_feat)

    local mixed_res = nn.JoinTable(2)({mid_feat, fusion_layer})

    local predict_YUV = upsample_and_color()(mixed_res)

    return nn.gModule({low_feat1, low_feat2}, {predict_YUV, predict_label})
  end


  return construct(256, 256)
end

return M
