import numpy as np
from Gradient import GradientInput,adam_iters
from joblib import Parallel,delayed
import time

class AdaptiveQaoaInput:

    def __init__(self,initial_pointR,n,pro_solve,qaoa_variant,sys_para):

        self.R=len(initial_pointR)
        
        self.initial_pointR=initial_pointR

        self.n=n

        self.qaoa_variant=qaoa_variant

        self.pro_solve=pro_solve

        self.sys_para=sys_para




class AdaptiveQaoaOutput:
        
    def __init__(self,energy,bias_energy,itera,bias,others):

        self.energy=energy

        self.bias_energy=bias_energy

        self.itera=itera

        self.bias=bias

        self.others=others
        
        

def adaptive_qaoa(input_para):

    n=input_para.n

    R_qaoa=input_para.R
    
    initial_pointR=input_para.initial_pointR.copy()
   
    best_bias_energy=np.zeros((2))      
    opt_bias_energyR=np.zeros((R_qaoa))
    best_bias=np.zeros(n)
    

    ave_iteration=np.zeros(1)
    opt_iterationR=np.zeros((R_qaoa))

    best_energy=np.zeros(1)
    opt_energyR=np.zeros((R_qaoa))

    best_other_opt=np.empty(1,dtype=object)

    t1=time.perf_counter()


    adam_input=[GradientInput(
                              initial_pointR[k],
                              n,
                              input_para.qaoa_variant,
                              input_para.pro_solve,
                              input_para.sys_para,
                              ) 
                     for k in range(R_qaoa)]
            
        
    #output
    if R_qaoa!=1:
        gradient_result=Parallel(n_jobs=10)(delayed(adam_iters)(adam) for adam in adam_input)
    else:
        gradient_result=[adam_iters(adam) for adam in adam_input]
    
    gradient_result=[gradient_result[k] for k in range(R_qaoa)]

    for k in range(R_qaoa):
        
        opt_energyR[k]=gradient_result[k].energy/(-1)
        opt_iterationR[k]=gradient_result[k].itera
        opt_bias_energyR[k]=gradient_result[k].bias_energy
   
    ma=np.where(opt_energyR==np.max(opt_energyR))
  
    p_ma=ma[0][0]
  
    best_energy=opt_energyR[p_ma]

    ave_iteration=np.mean(opt_iterationR,axis=0)
    
        
    best_bias=gradient_result[p_ma].bias_opt
    best_bias_energy[0]=gradient_result[p_ma].bias_energy
    best_bias_energy[1]=np.max(opt_bias_energyR)


    best_other_opt=gradient_result[p_ma].others

    t2=time.perf_counter()
    print('Type:',input_para.qaoa_variant['type'],',energy:',best_energy,',bias_energy:',best_bias_energy,',warm-start energy:',input_para.qaoa_variant['energy_WS'],',iterations:',ave_iteration,',time',t2-t1)
    t1=t2
        
    result = AdaptiveQaoaOutput(best_energy,
                                best_bias_energy,
                                ave_iteration,
                                best_bias,
                                best_other_opt)

    
    return result

