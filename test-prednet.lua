-- Eugenio Culurciello, SangPil Kim
-- August - September 2016
-- PredNet in Torch7 - from: https://arxiv.org/abs/1605.08104
-------------------------------------------------------------------------------

require 'prednet'
local c = require 'trepl.colorize'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- model parameters:
local opt = {}
opt.nlayers = 1
opt.inputSizeW = 64
opt.stride = 1
opt.poolsize = 2
opt.nFilters = {1, 32, 64, 128}
local clOpt = {}
clOpt['nSeq'] = 19
clOpt['kw'] = 3
clOpt['kh'] = 3
clOpt['st'] = 1
clOpt['pa'] = 1
clOpt['dropOut'] = 0
clOpt['lm'] = 1

-- instantiate MatchNet:
print('Creating model')
local model = PredNet(opt.nlayers, opt.stride, opt.poolsize, opt.nFilters, clOpt, true)

-- test:
print('Testing model')
local inTable = {}
table.insert( inTable, torch.ones(opt.nFilters[1], opt.inputSizeW, opt.inputSizeW)) -- input
for L=1, opt.nlayers do
   table.insert( inTable, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- E(t-1)
   table.insert( inTable, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- C(t-1)
   table.insert( inTable, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- H(t-1)
end

local outTable = model:forward(inTable)
print('Output is: ')
print(outTable)
graph.dot(model.fg, 'PredNet','Model') -- graph the model!
