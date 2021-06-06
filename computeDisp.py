import numpy as np
import cv2.ximgproc as xip
import cv2

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    #labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    #print(Il.shape)
    Il_pad = np.pad(Il,((1,1),(1,1),(0,0)))
    #print(Il_pad.shape)
    #print(Ir.shape)
    Ir_pad = np.pad(Ir,((1,1),(1,1),(0,0)))
    #print(Ir_pad.shape)
    cost_box = np.zeros((h, w, ch, max_disp), dtype=np.float32)
    
    position = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]

    Ir_census = np.zeros((h, w, ch,8), dtype=np.bool)
    for i in range(len(position)):
        posrow,poscol = position[i]
        Ir_census[:,:,:,i] = Ir<Ir_pad[posrow:posrow+h,poscol:poscol+w,:]
    #print(Ir_census.shape)
    Il_census = np.zeros((h, w, ch,8), dtype=np.bool)
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
    JBF_d = 5
    JBF_c = 0.1
    JBF_s = 2
    for disp in range(max_disp):
        cost_box[:,:,:,disp] = xip.jointBilateralFilter(Il, cost_box[:,:,:,disp], JBF_d, JBF_c, JBF_s)
    #print(cost_box.shape)
    cost_box = np.sum(cost_box,axis=2)
    #print(cost_box.shape)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    labels = np.argmin(cost_box,axis=2).astype(np.uint8)
    #print(labels.shape)

    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    #compute disparity map for right image
    right_cost_box = np.zeros((h, w, ch, max_disp), dtype=np.float32)
    for disp in range(max_disp):
        temp_censuscost = np.logical_xor(Il_census[:,disp:,:,:],Ir_census[:,:w-disp,:,:])
        temp_censuscost = np.sum(temp_censuscost,axis=3)
        #print(temp_censuscost.shape)
        temp_censuscost = np.pad(temp_censuscost,((0,0),(disp,0),(0,0),),mode='edge')
        #print(temp_censuscost.shape)
        right_cost_box[:,:,:,disp] = temp_censuscost

    # Weighted median filtering
    Il_gray = cv2.cvtColor(Il,cv2.COLOR_BGR2GRAY).astype(np.uint8)
    labels = xip.weightedMedianFilter(Il_gray,labels,2)


    return labels.astype(np.uint8)
    