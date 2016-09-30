-- Eugenio Culurciello, SangPil Kim
-- August - September 2016
-- PredNet in Torch7 - from: https://arxiv.org/abs/1605.08104
-------------------------------------------------------------------------------

require 'nn'
require 'prednet'
-- nngraph.setDebug(true)

local clOpt = {}
clOpt['nSeq'] = opt.nSeq
clOpt['kw'] = 3
clOpt['kh'] = 3
clOpt['st'] = opt.stride
clOpt['pa'] = opt.padding
clOpt['dropOut'] = 0
clOpt['lm'] = opt.lstmLayers

-- instantiate PredNet:
local unit = PredNet(opt.nlayers, opt.stride, opt.poolsize, opt.nFilters, clOpt, false) -- false testing mode
-- nngraph.annotateNodes()
-- graph.dot(unit.fg, 'PredNet-unit','Model-unit') -- graph the model!


-- clone model through time-steps:
local clones = {}
for i = 1, opt.nSeq do
   clones[i] = unit:clone('weight','bias','gradWeight','gradBias')
end


-- create model by connecting clones outputs and setting up global input:
-- inspired by: http://kbullaughey.github.io/lstm-play/rnn/
local E, C, H, E0, C0, H0, tUnit, P, xii, uInputs
E={} C={} H={} E0={} C0={} H0={} P={}
-- initialize inputs:
local xi = nn.Identity()()
for L=1, opt.nlayers do
   E0[L] = nn.Identity()()
   C0[L] = nn.Identity()()
   H0[L] = nn.Identity()()
   E[L] = E0[L]
   C[L] = C0[L]
   H[L] = H0[L]
end
-- create model as combination of units:
for i=1, opt.nSeq do
   -- set inputs to clones:
   uInputs={}
   xii = {xi} - nn.SelectTable(i) -- select i-th input from sequence
   table.insert(uInputs, xii)
   for L=1, opt.nlayers do
      table.insert(uInputs, E[L])
      table.insert(uInputs, C[L])
      table.insert(uInputs, H[L])
   end
   -- clones inputs = {input_sequence, E_layer_1, R_layer_1, E_layer_2, R_layer_2, ...}
   tUnit = clones[i] ({ table.unpack(uInputs) }) -- inputs applied to clones
   -- connect clones:
   for L=1, opt.nlayers do
      if i < opt.nSeq then
         E[L] = { tUnit } - nn.SelectTable(4*L-3) -- connect output E to prev E of next clone
         C[L] = { tUnit } - nn.SelectTable(4*L-2) -- connect output C to same layer C of next clone
         H[L] = { tUnit } - nn.SelectTable(4*L-1) -- connect output H to same layer H of next clone
      else
         P[L] = { tUnit } - nn.SelectTable(4*L) -- select Ah output as output of network
      end
   end
end
local inputs = {}
local outputs = {}
for L=1, opt.nlayers do
   table.insert(inputs, E0[L])
   table.insert(inputs, C0[L])
   table.insert(inputs, H0[L])
   table.insert(outputs, P[L])
end
table.insert(inputs, xi)
if opt.nlayers > 1 then
   outputs = {outputs-nn.SelectTable(1)}
end
model = nn.gModule(inputs, outputs ) -- output is P_layer_1 (prediction / Ah)
-- nngraph.annotateNodes()
-- graph.dot(model.fg, 'PredNet','Model') -- graph the model!


-- test overall model
print('Testing model')
local inTable = {}
local inSeqTable = {}
for i = 1, opt.nSeq do table.insert( inSeqTable,  torch.ones( opt.nFilters[1], opt.inputSizeW, opt.inputSizeW) ) end -- input sequence
for L=1, opt.nlayers do
   table.insert( inTable, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- E(t-1)
   table.insert( inTable, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- C(t-1)
   table.insert( inTable, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)))-- H(t-1)
end
table.insert( inTable,  inSeqTable ) -- input sequence
local outTable = model:updateOutput(inTable)
print('Model output is: ', outTable:size())
-- graph.dot(model.fg, 'PredNet','Model') -- graph the model!


criterion = nn.MSECriterion()
