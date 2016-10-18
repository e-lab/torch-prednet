-- First written by Sangpil Kim
-- Notation is from https://github.com/oxford-cs-ml-2015/practical6/blob/master/LSTM.lua
-- ConvLSTM with nngraph
-- August 2016

require 'nn'
require 'nngraph'

local sc = nn.SpatialConvolution
local scNB = nn.SpatialConvolution:noBias()
local sg = nn.Sigmoid

function convLSTM(inDim, outDim, opt)
  local dropout = opt.dropOut or 0
  local kw, kh  = opt.kw, opt.kh
  local stw, sth = opt.st, opt.st
  local paw, pah = opt.pa, opt.pa
  local n = opt.lm
  -- Input  is 1+ 2*#Layer
  -- Output is 1+ 2*#Layer
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- X
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- Cell
    table.insert(inputs, nn.Identity()()) -- Hidden state
  end

  local x
  local outputs = {}
  for L = 1,n do
     -- Container for previous C and H
    local prevH = inputs[L*2+1]
    local prevC = inputs[L*2]
    -- Get input
    if L == 1 then
      x = inputs[1]
    else
    -- Get x from bottom layer as input
      x = outputs[(L-1)*2]
    end
    --Convolutions
    local i2Ig, i2Fg, i2Og, i2It
    if L == 1 then
       i2Ig = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2Fg = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2Og = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2It = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x)
    else
       i2Ig = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2Fg = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2Og = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2It = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
    end

    local h2Ig = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevH)
    local h2Fg = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevH)
    local h2Og = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevC)
    local h2It = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevC)

    local ig = nn.CAddTable(1,1)({i2Ig, h2Ig})
    local fg = nn.CAddTable(1,1)({i2Fg, h2Fg})
    local og = nn.CAddTable(1,1)({i2Og, h2Og})
    local it = nn.CAddTable(1,1)({i2It, h2It})

    -- Gates
    local inGate = sg()(ig)
    local fgGate = sg()(fg)
    local ouGate = sg()(og)
    local inTanh = nn.Tanh()(it)
    -- Calculate Cell state
    local nextC = nn.CAddTable(1,1)({
        nn.CMulTable(1,1)({fgGate, prevC}),
        nn.CMulTable(1,1)({inGate, inTanh})
      })
    -- Calculate output
    local out = nn.CMulTable(1,1)({ouGate, nn.Tanh()(nextC)})

    table.insert(outputs, nextC)
   -- Dropout if neccessary
   if dropout > 0 then out = nn.Dropout(dropout)(nextH) end
    table.insert(outputs, out)
  end

  -- Extract output
  local lastH = outputs[#outputs]

  return nn.gModule(inputs, outputs)
end


