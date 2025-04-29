import numpy as np
import networkx as nx
from itertools import combinations as com
from scipy.special import comb


sigmax=np.array([[0,1],[1,0]],dtype=np.float64)
sigmay=np.array([[0,-1j],[1j,0]],dtype='complex128')
sigmaz=np.array([[1,0],[0,-1]],dtype=np.float64)
sigmai=np.array([[1,0],[0,1]],dtype=np.float64)


Problem_list=['MaxCut']
Problem=Problem_list[0]

Optimization_list=['Random']
Optimization_method=Optimization_list[0]


qubits=100

R_qaoa=10
R_gw=1
R_ws=10


seed=10#R_gw

learning_rate=0.4
epsilon_ws=0.25
max_iters=4000

np.random.seed(seed)

r=3

realizations=40


weights=np.ones(shape=(realizations,int(r*qubits/2)))

    
pro_info=np.zeros(shape=(realizations,int(r*qubits/2),3))


for i in range(realizations):
    
    gi=nx.Graph()
    gi=nx.random_regular_graph(r,qubits,i)
    
    j=0
    for u,v in gi.edges():
        pro_info[i][j][0]=u
        pro_info[i][j][1]=v
        
        pro_info[i][j][2]=weights[i][j]

        j=j+1


for i in range(1):#realizations):
    for j in range(int(r*qubits/2)):
        pro_info[i][j]=[np.min(pro_info[i][j][:2]),np.max(pro_info[i][j][:2]),pro_info[i][j][2]]
        
    p=np.argsort(pro_info[i][:,0])
    pro_info[i]=pro_info[i][p]
    
    v1=np.unique(pro_info[i][:,0])

    for k in range(len(v1)):
        p1=np.where(pro_info[i][:,0]==v1[k])[0]
 
        p2=np.argsort(pro_info[i][p1][:,1])

        pro_info[i][p1]=pro_info[i][p1][p2]
    
 


