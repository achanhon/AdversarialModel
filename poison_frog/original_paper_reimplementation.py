import torch


def find_candidate_for_collision(target, batchprovider, encoder,radius=3.0 / 255, alpha=1.0 / 255, iters=40):
    net.cuda()
    
    xt,yt = target
    if len(xt.shape)==3:
		xt = xt.view(1,xt.shape[0],xt.shape[1],xt.shape[2])
	with torch.no_grad():
		zt = encoder(xt)
    
    candidate,candidateafterattack,candidateclass=None,None,None
	for x, y in batchprovider:
		x, y = x.cuda(), y.cuda()
		
		zzt = torch.cat([zt]*x.shape[0],dim=0)
		
		xx = x.clone()
		for i in range(iters):
			x.requires_grad = True
			opt = torch.optim.SGD([x], lr=1)

			gap = (net(x)-zzt).abs()
			assert(len(gap.shape)==2)
			gap = torch.sum(gap*gap,dim=1)
			
			opt.zero_grad()
			loss.backward()

			adv_x = x + alpha * x.grad.sign()
			eta = torch.clamp(adv_x - original_x, min=-radius, max=radius)
			x = (original_x + eta).detach_()
