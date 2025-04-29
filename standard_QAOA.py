import numpy as np
from scipy.optimize import minimize
from global_var import *
from QAOA import construct_graph

    
def QAOA_e1(para,args):
    
    gamma1=para[0]
    beta1=para[1]
    
    n=args['n']
    
    e=np.zeros(3)    
    
    e[2]=-np.cos(gamma1)**2*np.sin(gamma1)*np.sin(4*beta1)
    
    e[1]=2*np.cos(gamma1)**2*np.sin(2*beta1)*np.sin(gamma1)
    e[1]=e[1]*(np.cos(beta1)*np.sin(beta1)*np.sin(gamma1)-np.cos(2*beta1))
    
    e[0]=2*np.cos(gamma1)**2*np.sin(2*beta1)*np.sin(gamma1)
    e[0]=e[0]*(np.sin(2*beta1)*np.sin(gamma1)-np.cos(2*beta1))
    
    E=np.sum(n*(1-e)/2)
    
    return -E

def QAOA_opt(graph_info,R=1,seed=10):
    
    np.random.seed(seed)
    n_sub=np.zeros(3)
    ER=np.zeros(R)
    pointR=np.zeros((R,2))
    
    for j,k,w in graph_info:
        
        edges_new,j_new,k_new,old_nodes,old_edges=construct_graph(graph_info,int(j),int(k))
 
        n_new=len(old_nodes)
        
        n_sub[n_new-4]+=1
        

    args={'n':n_sub}
   
 
    para=np.array([[4.23151749, 1.00024373]])



    for i in range(R):
        
        res=minimize(QAOA_e1, para[i],args=args,method='BFGS')
        ER[i]=-res['fun']
        pointR[i]=res['x']
        
    p=np.where(ER==np.max(ER))[0]
 
    E_opt=ER[p[0]]
    point_opt=pointR[p[0]]
    

    return E_opt,point_opt
    

e=np.zeros(realizations)
    
for i in range(realizations):
    print(qubits,i)
    e[i],b=QAOA_opt(pro_info[i],R=1)
    

    

    

    
    
    