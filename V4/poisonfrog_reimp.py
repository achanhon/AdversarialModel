import torch


def find_candidate_for_collision(sampleprovider, encoder,xt,yt,radius=3.0 / 255):
    net.cpu()
    with torch.no_grad():
        zt = encoder(xt)
        assert(len(zt.shape)==2)
    
    bestgap,candidate,candidateafterattack,candidateclass=None,None,None,None
    for x, y in batchprovider:
        if y==yt:
            continue
        assert(x.shape[0]==1) 
        
    
        x.requires_grad = True
        opt = torch.optim.SGD([x], lr=1)
        z = net(x)
            
        gap = torch.sum((z-zt).abs())
        opt.zero_grad()
        gap.backward()

        adv_x = x + radius * x.grad.sign()
        adv_x = torch.clamp(adv_x - original_x, min=0, max=1)
        eta = torch.clamp(adv_x - original_x, min=-radius, max=radius)
        adv_x = (x + eta).detach_()
        
        with torch.no_grad():
            z = net(adv_x)
            gap = torch.sum((z-zt).abs())
        
        if bestgap is None or gap<bestgap:
            candidate,candidateafterattack,candidateclass=x,adv_x,y
            
    return candidate,candidateafterattack,candidateclass


def eval_poisonfrog(testsampleprovider, nbtest, trainsampleprovider, encoder,xt,yt,radius=3.0 / 255):
    
