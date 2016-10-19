-- SangPil Kim, Eugenio Culurciello
-- August - September 2016
-------------------------------------------------------------------------------
local prevLoss = 1e10

function train(opt,datasetSeq, epoch, trainLog)

   print('==> training model')
   print  ('Loaded ' .. datasetSeq:size() .. ' images')
   model:training()
   local w, dE_dw = model:getParameters()
   print('Number of parameters ' .. w:nElement())
   print('Number of grads ' .. dE_dw:nElement())

   local cerr, ferr, loss= 0, 0, 0
   -- set training iterations and epochs according to dataset size:
   print('Training epoch #', epoch)

   local iteartion
   if opt.iteration == 0 then
      iteration = math.floor(datasetSeq:size()/opt.batch)
   else
      iteration = opt.iteration
   end
   for t = 1, iteration do
      xlua.progress(t, iteration)
      -- define eval closure
      local eval_E = function(w)

         model:zeroGradParameters()
         local sample = datasetSeq[t]
         local inTableG0, targetC, targetF = prepareData(opt,sample)
         --Get output
         -- 1st term is 1st layer of Ahat 2end term is 1stLayer Error
         local output = model:forward(inTableG0)
         -- Criterion is embedded
         -- estimate f and gradients
         -- Update Grad input
         local dE_dy = prepareDedw(output, targetF)
         model:backward(inTableG0,dE_dy)

         -- Display and Save picts
         if math.fmod(t*opt.batch, opt.disFreq) == 0 then
            local disFlag = 'train'
            display(opt, inTableG0[#inTableG0], targetF, targetC, output[1],disFlag)
            savePics(opt,targetF,output[1],epoch,t, disFlag)
         end
         --Calculate Matric
         -- Calculate Error and sum
         local tcerr , tferr , f = computMatric(targetC, targetF, output)
         cerr = cerr + tcerr
         ferr = ferr + tferr
         -- return f and df/dw
         return f, dE_dw
      end
      --Update model
      local _,fs = optim.adam(eval_E, w, optimState)
      -- compute statistics / report error
      loss = loss + fs[1]
      --------------------------------------------------------------------
   end
   if prevLoss > loss then prevLoss = loss end
   -- Save model
   if math.fmod(epoch, opt.saveEpoch) == 0 and prevLoss == loss then
      save(model, optimState, opt, epoch)
   end
   --Average errors
   --Batch is not divided since it is calcuated already in criterion
   cerr = cerr/iteration/opt.batch
   ferr = ferr/iteration/opt.batch
   loss = loss/iteration/opt.batch
   writLog(cerr,ferr,loss,trainLog)
   print('Learning Rate: ',optimState.learningRate)
   print ('Training completed!')
end
