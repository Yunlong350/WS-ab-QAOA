import numpy as np
from WSQAOA import QAOA



def gradient_function(para,qaoa,epsilon):
    
    e1=qaoa.get_expectation(para)
    
    gra=np.zeros(len(para))
    
    for j in range(len(para)):
        p_j=para[j].copy()
        para[j]=para[j]+epsilon   

        e2=qaoa.get_expectation(para)
        para[j]=p_j
        gra[j]=(e2-e1)/epsilon
    

    return gra,e1


class GradientInput:
    
    def __init__(self,para,n,qaoa_variant,pro_solve,sys_para):
        self.para=para

        self.n=n
        
        self.qaoa_variant=qaoa_variant

        self.pro_solve=pro_solve

        self.max_iters=sys_para['max_iters']



        
class GradientOutput:

    def __init__(self,para_opt,energy,itera,bias_energy,bias_opt,others):

        self.para_opt=para_opt

        self.energy=energy
 
        self.itera=itera

        self.bias_energy=bias_energy

        self.bias_opt=bias_opt

        self.others=others




def adam_iters(gradient):

    para_precision=1e-5

    final_precision=1e-3

    theta_precision=1e-1
    
    beta1=0.9
    beta2=0.999
    alpha=0.02
    epsilon=10**(-8)   
    
    npoints=2
    


    m=np.zeros(npoints,dtype=np.float64)
    v=np.zeros(npoints,dtype=np.float64)   

    mh=np.zeros(gradient.n,dtype=np.float64)
    vh=np.zeros(gradient.n,dtype=np.float64)


    qaoa=QAOA(gradient.n,
              gradient.qaoa_variant,
              gradient.pro_solve,
                )
    
    theta=np.zeros(shape=(int(gradient.max_iters+1),npoints),dtype=np.float64)
    theta[0]=gradient.para.copy() 
    bias=np.zeros(shape=(int(gradient.max_iters+1),gradient.n),dtype=np.float64)


    energy=np.zeros(int(gradient.max_iters+1),dtype=np.float64)
    bias_energy=np.zeros(int(gradient.max_iters+1))
    
    g=np.zeros(shape=(int(gradient.max_iters+1),npoints),dtype=np.float64)


    g[0],energy[0]=gradient_function(theta[0],qaoa,para_precision)  

    bias[0]=qaoa.bias
    bias_energy[0]=qaoa.get_bias_energy()

    t=0
    for t in range(1,gradient.max_iters+1):
        
        m=beta1*m+(1-beta1)*g[t-1]
        v=beta2*v+(1-beta2)*g[t-1]**2
        
        alphat=alpha*np.sqrt(1-beta2**t)/(1-beta1**t)
        
        
        theta_change=alphat*m/(np.sqrt(v)+epsilon)
        theta[t]=theta[t-1]-theta_change

        
        if qaoa.variant['type']=='ab':

            alphah=qaoa.variant['ell']*np.sqrt(1-beta2**t)/(1-beta1**t)
            mh,vh=qaoa.update_bias(mh,vh,alphah,theta[t-1])                       

    
        bias_energy[t]=qaoa.get_bias_energy()
        bias[t]=qaoa.bias

        g[t],energy[t]=gradient_function(theta[t],qaoa,para_precision)

        
        if np.abs(energy[t]-energy[t-1]) < final_precision and np.linalg.norm(theta_change)<theta_precision\
            and np.linalg.norm(bias[t]-bias[t-1])<theta_precision: 
            
            break

    other={'theta':theta[:t+1],'energy_opt':energy[:t+1],'bias_energy':bias_energy[:t+1],'bias_opt':bias[:t+1]}  

    return GradientOutput(theta[t],energy[t],t+1,bias_energy[t],bias[t],other)

    
   

  

    
        
