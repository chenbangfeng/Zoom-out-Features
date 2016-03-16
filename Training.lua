opt = {}
opt.learningRate = params.learningRate
opt.batchSize =  params.batchSize
opt.weightDecay = params.weightDecay
opt.momentum = params.momentum
opt.saveflag = params.saveflag
opt.save = params.save -- subdirectory to save model in
opt.type = params.dataType

--make a confusion matrix
categories = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21'}
confusion = optim.ConfusionMatrix(categories) 

optimState = nil
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  dampening = 0.0,
  learningRateDecay = 0
}
optimMethod = optim.sgd

if model then
	parameters,gradParameters = model:getParameters()
end

function train() -- A function that handles the training and report the accuracy on the training set
	epoch = epoch or 1 --epoch tracker
	local time = sys.clock() --local vars
	model:training() --set model to training
	--shuffle = torch.randperm(trainData.size) -- shuffle indices  at each epoch
	--print('==> doing epoch on training data:')
	--print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,trainData.size do
		xlua.progress(t, trainData.size) --display progress
		
		--load data
		if params.traintype == -1 then
			inputs,targets = load_minibatch(t,epoch,3)
		else
			inputs = load_image(trainData.inputs[t])
			targets = load_GT(trainData.targets[t])
		end
		--change data type
		if opt.type == 'double' then 
			inputs = inputs:double()
			targets = targets:double()
        elseif opt.type == 'cuda' then 
			inputs = inputs:cuda()
			targets = targets:cuda()			
		else 
			inputs = inputs:float() 
			targets = targets:float()
		end

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end
			-- reset gradients
			gradParameters:zero()
			
			-- f is the average of all criterions
			local f = 0
			
			-- evaluate function for complete mini batch
			output = model:forward(inputs);
			--local _,output = yhat:max(2)
			f = criterion:forward(output,targets);
			print(f)
			local df_do = criterion:backward(output, targets);
			model:backward(inputs, df_do);
			--for i = 1,inputs:size()[1] do
			--confusion:add(output[i]:squeeze(),targets[i]:squeeze())			
			--end
			return f,gradParameters
		end
		-- optimize on current mini-batch
		if optimMethod == optim.asgd then
			_,_,average = optimMethod(feval, parameters, optimState)
		else
			optimMethod(feval, parameters, optimState)
		end
	end
	time = sys.clock() - time  --time taken
	time = time / trainData.size
	--print(confusion)
	--print(feval)
epoch = epoch + 1
confusion:zero()
end



