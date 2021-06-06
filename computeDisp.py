import numpy as np
import cv2.ximgproc as xip
import cv2

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    #labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    pad_num = 6
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    #print(Il.shape)
    Il_pad = np.pad(Il,((pad_num,pad_num),(pad_num,pad_num),(0,0)))
    #print(Il_pad.shape)
    #print(Ir.shape)
    Ir_pad = np.pad(Ir,((pad_num,pad_num),(pad_num,pad_num),(0,0)))
    #print(Ir_pad.shape)
    cost_box = np.zeros((h, w, ch, max_disp), dtype=np.float32)
    
    position = []
    for i in range(0,1+pad_num*2):
        for j in range(0,1+pad_num*2):
            if i!=pad_num or j!=pad_num:
                position.append((i,j))

    Ir_census = np.zeros((h, w, ch,len(position)), dtype=np.bool)
    for i in range(len(position)):
        posrow,poscol = position[i]
        Ir_census[:,:,:,i] = Ir<Ir_pad[posrow:posrow+h,poscol:poscol+w,:]
    #print(Ir_census.shape)
    Il_census = np.zeros((h, w, ch,len(position)), dtype=np.bool)
    for i in range(len(position)):
        posrow,poscol = position[i]
        Il_census[:,:,:,i] = Il<Il_pad[posrow:posrow+h,poscol:poscol+w,:]

    for disp in range(max_disp):
        temp_censuscost = np.logical_xor(Il_census[:,disp:,:,:],Ir_census[:,:w-disp,:,:])
        temp_censuscost = np.sum(temp_censuscost,axis=3)
        #print(temp_censuscost.shape)
        temp_censuscost = np.pad(temp_censuscost,((0,0),(disp,0),(0,0),),mode='edge')
        #print(temp_censuscost.shape)
        cost_box[:,:,:,disp] = temp_censuscost


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    JBF_s = 5
    JBF_d = 4*JBF_s
    JBF_c = 0.1
    
    for disp in range(max_disp):
        cost_box[:,:,:,disp] = xip.jointBilateralFilter(Il, cost_box[:,:,:,disp], JBF_d, JBF_c, JBF_s)
    #print(cost_box.shape)
    cost_box = np.sum(cost_box,axis=2)
    #print(cost_box.shape)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    left_disparity = np.argmin(cost_box,axis=2).astype(np.uint8)
    #print(labels.shape)

    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    # compute disparity map for right image
    right_cost_box = np.zeros((h, w, ch, max_disp), dtype=np.float32)
    for disp in range(max_disp):
        temp_censuscost = np.logical_xor(Il_census[:,disp:,:,:],Ir_census[:,:w-disp,:,:])
        temp_censuscost = np.sum(temp_censuscost,axis=3)
        temp_censuscost = np.pad(temp_censuscost,((0,0),(0,disp),(0,0),),mode='edge')
        right_cost_box[:,:,:,disp] = temp_censuscost
    for disp in range(max_disp):
        right_cost_box[:,:,:,disp] = xip.jointBilateralFilter(Ir, right_cost_box[:,:,:,disp], JBF_d, JBF_c, JBF_s)
    right_cost_box = np.sum(right_cost_box,axis=2)
    right_disparity = np.argmin(right_cost_box,axis=2).astype(np.uint8)
    
    # check consistency
    wv, hv = np.meshgrid(range(w), range(h))
    #print(hv.shape)
    #print(wv.shape)
    R_idx = np.stack((wv, hv))
    #print(left_disparity.shape)
    #print(R_idx[0,:,:].shape)
    R_idx[0,:,:] = R_idx[0,:,:] - left_disparity
    R_idx = np.reshape(R_idx,(2,h*w))
    #print(R_idx.shape)
    DR_x_DL = right_disparity[R_idx[1,:],R_idx[0,:]]
    DR_x_DL = np.reshape(DR_x_DL,(h,w))
    #print(DR_x_DL.dtype)
    valid = left_disparity==DR_x_DL
    #print(valid.shape)
    
    # hole filling
    valid = np.pad(valid,((0,0),(1,1)),constant_values=True)
    left_disparity = np.pad(left_disparity,((0,0),(1,1)),constant_values=max_disp)
    #print(valid.shape)
    for iw in range(1,w+1):
        for ih in range(h):
            if valid[ih,iw] == False:
                find_x = iw
                while valid[ih,find_x]==False:
                    find_x = find_x - 1
                FL = left_disparity[ih,find_x]
                find_x = iw
                while valid[ih,find_x]==False:
                    find_x = find_x + 1
                FR = left_disparity[ih,find_x]
                left_disparity[ih,iw] = min(FL,FR)
    left_disparity = left_disparity[:,1:-1]
    
    # Weighted median filtering
    Il_gray = cv2.cvtColor(Il,cv2.COLOR_BGR2GRAY).astype(np.uint8)
    labels = xip.weightedMedianFilter(Il_gray,left_disparity,7)


    return labels.astype(np.uint8)
    