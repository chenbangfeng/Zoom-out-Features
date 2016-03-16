function load_image_names(filePath)
	local loaded = matio.load(filePath)
	im_list = loaded.Imlist or loaded.Imlistgt
	local t = {}
	local s = {}
	for i = 1, im_list:size()[1] do
		for j = 1, im_list:size()[2]  do
			t[j] = string.char(im_list[{i,j}])
  		end
  		s[i]=table.concat(t);
	end
	im_list = s --this list contains the names of the images
	return im_list	
end



function pre_process(input_image,Model)
	local prepModel = Model
	prepModel:evaluate()
	local mean = torch.Tensor({0.485,0.456,0.406})
	local std = torch.Tensor({0.229,0.224,0.225})
	local im = image.scale(input_image,224,224, 'bilinear')
	for channel = 1,3 do
		im[channel]:add(-mean[channel])
		im[channel]:div(std[channel])
	end
	return prepModel:forward(im:view(1,3,224,224):cuda()):float()
end




function load_image(image_id)
	if stats == nil then
		stats = torch.load('train_stats_5096.t7')
		stats.mean = stats.mean:view(1,5096,1,1):expand(1,5096,56,56)
		stats.std = stats.std:view(1,5096,1,1):expand(1,5096,56,56)
	end
	local im = image.load(image_id)
	im = pre_process(im,prepNet) --normalize, pass through resnet
	im = im:add(-stats.mean) --renormalize with dataset statistis
	im = im:cdiv(stats.std)
	return im
end




function load_GT(target_id,scale)
	local GT = matio.load(target_id).GT
	GT = GT:contiguous():view(1, GT:size()[1], GT:size()[2])
	if scale == 1 then
		GT = image.scale(GT, 56, 56, 'simple'):squeeze():view(1,56,56)
	end
	return GT
	
end




function load_minibatch(t,epoch,sample_size)
	local nsample = sample_size
	----
	if params.traintype == -1 then
		local im = load_image(trainData.inputs[t])
		local gt = load_GT(trainData.targets[t]):squeeze()
		
		--find the background pixels
		local inds = gt:eq(21):nonzero()
		local perm = torch.randperm(inds:size()[1])
		
		coords = torch.zeros(math.min(inds:size()[1], nsample),2)
		tars = torch.zeros(math.min(inds:size()[1], nsample))
		----
		for i = 1, math.min(inds:size()[1], nsample) do
			print(i)
			coords[i] = inds[perm[i]]	
			tars[i] = 21
		end	
		-----
		for j = 1,20 do
			local inds = gt:eq(j):nonzero()
			----
			if inds:dim() > 0 then 
				local perm = torch.randperm(inds:size()[1])
				local tmpc = torch.zeros(math.min(inds:size()[1], nsample),2)
				local tmpt = torch.zeros(math.min(inds:size()[1], nsample))
				----
				for i = 1, math.min(inds:size()[1], nsample) do
					tmpc[i] = inds[perm[i]]
					tmpt[i] = j
				end
				---
				coords = torch.cat(coords,tmpc,1)
				tars = torch.cat(tars,tmpt)
			end
			----
		end
		----
		local n = coords:size()[1]
		local ins = torch.Tensor(1,5096,n,1)
		local start = 1
		for j = start, n do
			ins[{1, {}, j, 1}] = im[{1, {}, coords[j][1], coords[j][2]}]
		end
		tars = tars:view(1,n,1)
		return ins, tars
	
	else
		ins = load_image(trainData.inputs[t])
		tars = load_GT(trainData.targets[t]):squeeze()
	end
	return ins, tars
end




function validate_model() 
	local valconfusion = optim.ConfusionMatrix(categories)
	----
	for i = 1,valData.size do
		--load image
		local valIm = load_image(valData.inputs[i])
		local valTar = load_GT(valData.targets[i])
		--process image
		pred = model:forward(valIm:cuda()):float():squeeze() --21x56x56
			--upsample the prediction to the size of the original image
		_,pred2 = torch.max(pred,1)
		valconfusion:batchAdd(pred2:squeeze():view(pred2:nElement()),valTar:squeeze():view(56*56))
	end
	----
	print(valconfusion)
	return valconfusion
end



function save_validation_results(scale)
	----
	for i = 1,valData.size do
		print(i)
		valIm = load_image(valData.inputs[i])
		valTar = load_GT(valData.targets[i],2)
		pred = model:forward(valIm:cuda()):float():squeeze()
		pred1 = image.scale(pred,valTar:size()[3], valTar:size()[2],'bilinear')
		_,pred2 = torch.max(pred1,1)
		matio.save('Results/'..i..'.mat', pred2:squeeze())
	end
	----
return
end




function generate_predictions()
	for i = 1,valData.size do
		--load image
		local im1 = image.load(valData.inputs[i])
		--process image
		local im2 = pre_process(im1, prepNet)
		--load targets
		local tar = matio.load(valData.targets[i])
		--make prediction on validation image
		local pred = model:forward(im2:cuda())
		--upsample the prediction
		local predd = image.scale(pred,im1:size()[2],im1:size()[3],'simple')
		local _,pred2 = torch.max(pred,1)	
		--save image
	end
	return
end



function choose_pixels(dataset,batchSize,t)
	local count = 0
	local targets = torch.Tensor(math.min(t+batchSize,dataset.size+1)-t,56,56):zero()
	for i = t,math.min(t+batchSize-1, dataset.size) do
		count=count+1
		local tar1 = matio.load(dataset.targets[shuffle[i]])
		--local tar2 = image.scale(tar1.GT,56,56,'bilinear')
		local tar3 = image.scale(tar1.GT,56,56,'simple')
		local classes = find_classes(tar3)
		targets[{count,{},{}}] = tar3 
	end
	return targets
end




function find_classes(target)
	local class_counts = torch.Tensor(21,1):zero()
	for i = 1,21 do
		if target[target:eq(i)]:sum() > 0 then
			class_counts[i] = i
		end
	end
	class_counts[class_counts:gt(0)] = 1
	return class_counts
end




function enumerate_classes(dataset)
	--This file takes in 'dataset' which is
	--a table containing the names of the .mat files. 
	--Input should be trainData.groundtruths
	count = torch.Tensor(21,1):zero()
	for i  = 1,#dataset do
		local im1 = matio.load(dataset[i])
		local im2 = image.scale(im1.GT,56,56,'simple')	
		count = count + find_classes(im2)
	end
	return count
end



























