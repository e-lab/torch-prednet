-- Eugenio Culurciello, Sangpil Kim (batch mode added)
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- convLSTM model from SangPil Kim

require 'nn'
require 'nngraph'
require 'models/convLSTM'
local c = require 'trepl.colorize'
--nngraph.setDebug(true)


function MatchNet(nlayers, input_stride, poolsize, mapss, clOpt, testing, batch)
   local pE, A, upR, C, H, Ah, E, R, cA, mA, cAh, up, op, convlstm
   E={} -- output from layers are saved to connect to next layer input
   C={} -- LSTM cell state
   H={} -- LSTM hidden state
   convlstm = {} -- convLSTM modules
   R = {} -- output from convLSTM modules

   -- Ah = prediction / generator branch, A_hat in paper, E = error output
   -- This module creates the MatchNet network model, defined as:
   -- inputs = { E(t-1), C(t-1), H(t-1) }, where t-1 = previous instant
   -- outputs = { E(t), C(t), H(t), Ah(t) }, where t = this instant

   -- creating input / output list:
   local inputs = {}
   local outputs = {}

   -- initializing inputs: 1 + 3*nlayers in total
   inputs[1] = nn.Identity()() -- global input
   for L = 1, nlayers do
      inputs[3*L-1] = nn.Identity()() -- E(t-1), layer error output
      inputs[ 3*L ] = nn.Identity()() -- C(t-1), LSTM cell state
      inputs[3*L+1] = nn.Identity()() -- H(t-1), LSTM hidden state
   end

   -- generating network layers (2 for loops):

   -- first recurrent branch needs to be updated from top:
   for L = nlayers,1,-1 do
      -- R / recurrent branch:
      up = nn.SpatialUpSamplingNearest(poolsize)
      if L == nlayers then
         -- create convLSTM modules:
         convlstm[L] = convLSTM(2*mapss[L], mapss[L], clOpt)
         -- input to convLSTM: E(t-1), C(t-1), H(t-1)
         R[L] = { inputs[3*L-1], inputs[3*L], inputs[3*L+1] } - convlstm[L]
      else
         -- create convLSTM modules:
         -- if L == 1 then
            -- convlstm[L] = convLSTM(4*mapss[L], mapss[L], clOpt)
         -- else
            convlstm[L] = convLSTM(mapss[L+1]+2*mapss[L], mapss[L], clOpt)
         -- end
         upR = R[L+1] - nn.SelectTable(2,2) - up -- select 2nd = LSTM cell state
         if batch > 1 then
            inR = { upR, inputs[3*L-1] } - nn.JoinTable(2,2*mapss[L]) -- join R(t) from upper layer and E(t-1)
         else
            inR = { upR, inputs[3*L-1] } - nn.JoinTable(1)-- join R(t) from upper layer and E(t-1)
         end
         R[L] = { inR, inputs[3*L], inputs[3*L+1] } - convlstm[L]
      end
      if testing then R[L]:annotate{graphAttributes = {color = 'red', fontcolor = 'black'}} end
   end

   -- the we update bottom-up discriminator and generator network:
   for L = 1, nlayers do
      if testing then print('MatchNet model: creating layer:', L) end

      -- A branch:
      if L == 1 then
         A = inputs[1] -- global input
      else
         pE = E[L-1] -- previous layer E
         cA = nn.SpatialConvolution(2*mapss[L-1], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
         mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
         A = pE - cA - mA - nn.ReLU()
      end
      if testing then A:annotate{graphAttributes = {color = 'green', fontcolor = 'black'}} end

      -- A-hat branch:
      cAh = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- Ah convolution
      iR = R[L] - nn.SelectTable(2,2) -- select 2nd = LSTM cell state
      if L == 1 then
         Ah = {iR} - cAh - nn.HardTanh(0,1) -- saturating ReLU like in original paper
      else
         Ah = {iR} - cAh - nn.ReLU()
      end
      op = nn.PReLU(mapss[L])

      -- E[L] = {A, Ah} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
      if batch > 1 then
         E[L] = { {A, Ah} - nn.CSubTable(1,1) - nn.ReLU(), {Ah, A} - nn.CSubTable(1,1) - nn.ReLU() } - nn.JoinTable(2,2*mapss[L]) -- same and PredNet model
      else
         E[L] = { {A, Ah} - nn.CSubTable(1,1) - nn.ReLU(), {Ah, A} - nn.CSubTable(1,1) - nn.ReLU() } - nn.JoinTable(1) -- same and PredNet model
      end
      if testing then E[L]:annotate{graphAttributes = {color = 'blue', fontcolor = 'black'}} end

      -- output list 4 for each layer, 4 * nlayers in total):
      outputs[4*L-3] = E[L] -- E(t)
      outputs[4*L-2] = R[L] - nn.SelectTable(1,1) -- C(t), LSTM cell state
      outputs[4*L-1] = R[L] - nn.SelectTable(2,2) -- H(t), LSTM hidden state
      outputs[4*L] = Ah -- prediction output (t)
   end

   local g = nn.gModule(inputs, outputs)
   if testing then nngraph.annotateNodes() end
   return g

end
