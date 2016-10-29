-- MatchNet training: predicting future frames in video
--Sangpil Kim, Eugenio Culurciello August - September 2016
--
-- code training and testing inspired by: https://github.com/viorik/ConvLSTM
--

require 'nn'
require 'models/matchnet'
-- nngraph.setDebug(true)
local class = require'class'
local S = class('S')
function S:__init(opt)
   for name, value in pairs(opt) do
      self[name] = value
   end
end
function S:getModel()
   if self.useGPU then
      require 'cudnn'
      require 'cunn'
      require 'cutorch'
   end
   -- load trained version of matchnet
   local unit = torch.load(paths.concat(self.model,'model.net')) -- false testing mode
   -- nngraph.annotateNodes()
   -- graph.dot(unit.fg, 'MatchNet-unit','Model-unit') -- graph the model!

   -- clone model through time-steps:
   local clones = {}
   for i = 1, self.nSeq do
      clones[i] = unit:clone('weight','bias','gradWeight','gradBias')
   end

   -- create model by connecting clones outputs and setting up global input:
   -- inspired by: http://kbullaughey.github.io/lstm-play/rnn/
   local E, C, H, E0, C0, H0, tUnit, P, xii, uInputs, eCon, cellCon, hiddenCon, pCon
   E={} C={} H={} E0={} C0={} H0={} P={} eCon={} cellCon={} hiddenCon={} pCon={}
   -- initialize inputs:
   local xi = nn.Identity()()
   for L=1, self.nlayers do
      E0[L] = nn.Identity()()
      C0[L] = nn.Identity()()
      H0[L] = nn.Identity()()
      E[L] = E0[L]
      C[L] = C0[L]
      H[L] = H0[L]
   end
   -- create model as combination of units:
   for i=1, self.nSeq do
      -- set inputs to clones:
      uInputs={}
      xii = {xi} - nn.SelectTable(i,i) -- select i-th input from sequence
      table.insert(uInputs, xii)
      for L=1, self.nlayers do
         table.insert(uInputs, E[L])
         table.insert(uInputs, C[L])
         table.insert(uInputs, H[L])
      end
      -- clones inputs = {input_sequence, E_layer_1, R_layer_1, E_layer_2, R_layer_2, ...}
      tUnit = clones[i] ({ table.unpack(uInputs) }) -- inputs applied to clones
      -- connect clones:
      for L=1, self.nlayers do
         if i < self.nSeq then
            E[L] = { tUnit } - nn.SelectTable(4*L-3,4*L-3) -- connect output E to prev E of next clone
            C[L] = { tUnit } - nn.SelectTable(4*L-2,4*L-2) -- connect output R to same layer E of next clone
            H[L] = { tUnit } - nn.SelectTable(4*L-1,4*L-1) -- connect output R to same layer E of next clone
            if L == 1 then
               table.insert(eCon, E[L])
            end
         else
            E[L] = { tUnit } - nn.SelectTable(4*L-3,4*L-3) -- connect output E to prev E of next clone
            P[L] = { tUnit } - nn.SelectTable(4*L,4*L) -- select Ah output as output of network
            table.insert(pCon, P[L])
            table.insert(eCon, E[L])
            if self.modelKeep then
               C[L] = { tUnit } - nn.SelectTable(4*L-2,4*L-2) -- connect output R to same layer E of next clone
               H[L] = { tUnit } - nn.SelectTable(4*L-1,4*L-1) -- connect output R to same layer E of next clone
               table.insert(cellCon, C[L]) -- connect output E to prev E of next clone
               table.insert(hiddenCon, H[L]) -- connect output E to prev E of next clone
            end
         end
      end
   end
   local inputs = {}
   local outputs = {}
   for L=1, self.nlayers do
      table.insert(inputs, E0[L])
      table.insert(inputs, C0[L])
      table.insert(inputs, H0[L])
   end
   table.insert(inputs, xi)
   function putTable(src,dst)
      for _, val in ipairs(src) do
         table.insert(dst,val)
      end
   end
   putTable(pCon,outputs)
   putTable(eCon,outputs)
   if self.modelKeep then
      putTable(cellCon,outputs)
      putTable(hiddenCon,outputs)
   end
   local model = nn.gModule(inputs, outputs ) -- output is P_layer_1 (prediction / Ah)
   if self.useGPU then
      model:cuda()
   end
   return model
end
return S
