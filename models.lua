function build_prepNet()
	--load the resnet model
	local filepath = '/home-nfs/jmichaux/Code/resnet-50.t7'
	local resnet = torch.load(filepath)
	resnet:evaluate()
	iminput = nn.Identity()()
	-- the input to the network is a 4D tensor of 3x224x224 images
	C1 = resnet:get(1)(iminput)
	C2 = resnet:get(2)(C1)
	C3 = resnet:get(3)(C2)
	C4 = resnet:get(4)(C3)
	C5 = resnet:get(5)(C4)
	C6 = resnet:get(6)(C5)
	C7 = resnet:get(7)(C6)
	C8 = resnet:get(8)(C7)
	C9 = resnet:get(9)(C8)
	C10 = resnet:get(10)(C9)
	C11 = resnet:get(11)(C10)
	
	C1_down = nn.SpatialMaxPooling(2,2,2,2):cuda()(C1)
	C2_down = nn.SpatialMaxPooling(2,2,2,2):cuda()(C2)
	C3_down = nn.SpatialMaxPooling(2,2,2,2):cuda()(C3)

	C6_scaled = nn.SpatialUpSamplingNearest(2):cuda()(C6)
	C7_scaled = nn.SpatialUpSamplingNearest(4):cuda()(C7)
	C8_scaled = nn.SpatialUpSamplingNearest(8):cuda()(C8)

	C11_rep = nn.Replicate(56,4)(nn.Replicate(56,3)(C11))	
	
	output = nn.JoinTable(2)({C1_down, C2_down, C3_down, C4, C5, C6_scaled, C7_scaled, C8_scaled, C11_rep})
	model1 = nn.gModule({iminput}, {output})
	return model1
end



function build_segNet()
	local model = nn.Sequential()
	local U1 = cudnn.SpatialConvolution(5096,256,1,1)
	local U2 = cudnn.SpatialConvolution(256,256,1,1)
	local U3 = cudnn.SpatialConvolution(256,21,1,1)
	model:add(U1)
	model:add(nn.ReLU())
	model:add(U2)
	model:add(nn.ReLU())
	model:add(U3)
	return model
end



function build_segNet3()
	local model = nn.Sequential()
	local u1 = cudnn.SpatialConvolution(5096,256,3,3,1,1,1,1)
	local u2 = cudnn.SpatialConvolution(256,256,3,3,1,1,1,1)
	local u3 = cudnn.SpatialConvolution(256,21,3,3,1,1,1,1)
	model:add(u1)
	model:add(nn.ReLU())
	model:add(u2)	
	model:add(nn.ReLU())
	model:add(u3)
	return model
end


















