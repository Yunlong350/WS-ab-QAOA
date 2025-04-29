import numpy as np
from Adaptive_para import AdaptiveQaoaInput,adaptive_qaoa
import time
from SDP import GW_algorithm,maxcut_interior_point
from BMMK import BMMK_optR
from global_var import *

start=time.perf_counter()



energy=np.zeros((3,realizations))    
bias_energy=np.zeros((realizations,2))  

ite=np.zeros((3,realizations))

bias=np.zeros(shape=(realizations,qubits))

GW_energy=np.zeros((realizations,3))
GW_solution=np.zeros((realizations,qubits))
GWE_total=np.zeros((realizations,R_gw))

BMMK_energy=np.zeros((realizations))
BMMK_solution=np.zeros((realizations,qubits))
BMMK_ite=np.zeros((realizations))

    
other=np.empty(shape=((realizations,3)),dtype='object')

    
np.random.seed(10)



for i in range(1):#realizations):
    print('qubits',qubits,i,Problem,'ab')
    
    
    primal,dual,X=maxcut_interior_point(pro_info[i])
    v_ran=np.random.uniform(-1,1,(R_gw,qubits))

    GW_energy[i][0],GW_solution[i],GWE_total[i]=GW_algorithm(pro_info[i],X[-1],v_ran)
    GW_energy[i][1]=primal[-1]
    GW_energy[i][2]=dual[-1]

    
    qaoa_variant={'type':'ab','ell':learning_rate,'solution_WS':GW_solution[i]/np.sqrt(3),'energy_WS':GW_energy[i]}    
    initial_pointR=np.array([[4.23151749, 1.00024373]]) 
    #initial_pointR=np.random.uniform(0,2*np.pi,(R_qaoa,2))     
       
    para=AdaptiveQaoaInput (
                            initial_pointR,
                            qubits,
                            pro_info[i],
                            qaoa_variant,
                            {'max_iters':max_iters}
                            )
    
    res=adaptive_qaoa(para)
    
    energy[0][i]=res.energy
    bias_energy[i]=res.bias_energy
    ite[0][i]=res.itera
    bias[i]=res.bias
    other[i][0]=res.others
    
    
    initial_pointR=np.random.uniform(0,2*np.pi,(R_qaoa,2))
    initial_pointR[0]=np.array([5.76654488, 4.48981206])
    
    print('warm-start',i)
    qaoa_variant={'type':'warm-start','epsilon_ws':epsilon_ws,'solution_WS':GW_solution[i],'energy_WS':GW_energy[i]}    
    
    para=AdaptiveQaoaInput (
                            initial_pointR,
                            qubits,
                            pro_info[i],
                            qaoa_variant,
                            {'max_iters':max_iters}
                            )
    
    res=adaptive_qaoa(para)
    energy[1][i]=res.energy
    ite[1][i]=res.itera
    other[i][1]=res.others
    
    #
    theta0=np.random.uniform(-np.pi,np.pi,(R_ws,qubits))
    
    
    BMMK_energy[i],BMMK_solution[i],BMMK_ite[i]=BMMK_optR(theta0,pro_info[i],max_iters,k=2)
    
    #uniform rotation
    #thetar=np.random.uniform(-np.pi,np.pi,1)+np.pi/2
    #s_ws=BMMK_solution[i]+thetar
    
    #vertex at top
    nv=np.random.choice([i for i in range(qubits)],1)
    s_ws=BMMK_solution[i]-BMMK_solution[i][nv]
    
    
    print('warmest',i)
    qaoa_variant={'type':'warmest','solution_WS':s_ws,'energy_WS':BMMK_energy[i]}  
    
    para=AdaptiveQaoaInput (
                            initial_pointR,
                            qubits,
                            pro_info[i],
                            qaoa_variant,
                            {'max_iters':max_iters}
                            )
    
    res=adaptive_qaoa(para)
    energy[2][i]=res.energy
    ite[2][i]=res.itera
    other[i][2]=res.others
