import numpy as np


def construct_graph(edges,j,k=-1):
    
    if k==-1:
        edges_j=[]
        for v1,v2,w in edges:
            if j==v1 or j==v2:
                edges_j.append([v1,v2,w])
        
        edges_j=np.array(edges_j)
        nodes_j=np.unique(edges_j[:,:2])
            
        edges_new=np.array(edges_j)
        
        n_new=len(nodes_j)
    
        for i in range(n_new):
            edges_new[np.where(edges_new[:,:2]==nodes_j[i])]=i
            
        j_new=np.where(nodes_j==j)[0][0]

        return edges_new,j_new ,nodes_j,edges_j

    else:
        edges_jk=[]
        for v1,v2,w in edges:
            if j==v1 or j==v2 or k==v1 or k==v2:
                edges_jk.append([v1,v2,w])
        
        edges_jk=np.array(edges_jk)
        nodes_jk=np.unique(edges_jk[:,:2])
        
        edges_new=np.array(edges_jk)

        
        n_new=len(nodes_jk)
    
        for i in range(n_new):
            edges_new[np.where(edges_new[:,:2]==nodes_jk[i])]=i
            
        j_new=np.where(nodes_jk==j)[0][0]
        k_new=np.where(nodes_jk==k)[0][0]
        
        return edges_new,j_new,k_new,nodes_jk,edges_jk
    
def node_connection(j,k,graph_info):

    if k!=-1:  
  
        
        aj,bj,j_nodes,cj=construct_graph(graph_info,j,-1)
        
        ak,bk,k_nodes,ck=construct_graph(graph_info,k,-1)

        jc=np.setdiff1d(j_nodes,[j,k])
        kc=np.setdiff1d(k_nodes,[j,k])
        
        s=1
    
        jc2=-1*np.ones(len(jc)+1)
        kc2=-1*np.ones(len(kc)+1)
        
        jc2[0]=j
        kc2[0]=k
        
        
        for i1 in range(len(jc)):
            
            for i2 in range(len(kc)):
                
                if jc[i1]==kc[i2]:
                    
                    jc2[s]=jc[i1]
                    kc2[s]=kc[i2]
                    s+=1
                
        jc2[s:]=np.setdiff1d(jc,jc2[:s])
        kc2[s:]=np.setdiff1d(kc,kc2[:s])
    
  
        nodes=np.array([jc2,kc2],dtype=np.int64)
    else:
        aj,bj,j_nodes,cj=construct_graph(graph_info,j,-1)
        jc=np.setdiff1d(j_nodes,[j]).astype(np.int64)
        nodes=np.concatenate(([j],jc),axis=0)
    
    return nodes 



def coeff_ZYX(alpha,beta1,j):
    
    CZj=1-2*np.cos(alpha[j])**2*np.sin(beta1)**2
    CYj=np.cos(alpha[j])*np.sin(2*beta1)    
    CXj=-np.sin(2*alpha[j])*np.sin(beta1)**2
    
    c=np.array([CZj,CYj,CXj])
    
    return c


def EZZ(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]

    e=np.cos(delta[j])*np.cos(delta[k])*h[j]*h[k]
    
    return e

def EZY(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]
    
    
    e=np.cos(gamma1)**2*np.sin(gamma1)*np.cos(delta[j])*np.sin(delta[k])*np.cos(delta[k2])*h[j]*h[k]*h[k2]

    e+=np.cos(gamma1)**2*np.sin(gamma1)*np.cos(delta[j])*np.sin(delta[k])*np.cos(delta[k1])*h[j]*h[k]*h[k1]

    e+=np.cos(gamma1)**2*np.sin(gamma1)*np.sin(delta[k])*h[k]

    e-=np.sin(gamma1)**3*np.sin(delta[k])*np.cos(delta[k1])*np.cos(delta[k2])*h[k]*h[k1]*h[k2]

    return e

def EZX(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]
    
    e=np.cos(delta[j])*np.sin(delta[k])*np.cos(gamma1)**3*h[j]*h[k]
    
    e-=np.cos(delta[j])*np.sin(delta[k])*np.cos(delta[k1])*np.cos(delta[k2])*np.cos(gamma1)*np.sin(gamma1)**2*h[j]*h[k]*h[k1]*h[k2]
    
    e-=np.sin(delta[k])*np.cos(delta[k1])*np.cos(gamma1)*np.sin(gamma1)**2*h[k]*h[k1]
    
    e-=np.sin(delta[k])*np.cos(delta[k2])*np.cos(gamma1)*np.sin(gamma1)**2*h[k]*h[k2]
    
    return e
    
def EYY(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]
    
    e=np.cos(delta[j2])*np.cos(delta[k1])*h[j2]*h[k1]

    if j2==k2:
    
        e=e+1
    else:
        e=e+np.cos(delta[j2])*np.cos(delta[k2])*h[j2]*h[k2]

    e=e+np.cos(delta[j1])*np.cos(delta[k2])*h[j1]*h[k2]

    if j1==k1:
    
        e=e+1
    else:
        e=e+np.cos(delta[j1])*np.cos(delta[k1])*h[j1]*h[k1]
    
    e=np.cos(gamma1)**2*np.sin(gamma1)**2*np.sin(delta[j])*np.sin(delta[k])*h[j]*h[k]*e
    

    return e


def EXX(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]
    
    
    e=np.cos(gamma1)**4

    e=e-np.cos(gamma1)**2*np.sin(gamma1)**2*np.cos(delta[k1])*np.cos(delta[k2])*h[k1]*h[k2]
    
    e=e-np.cos(gamma1)**2*np.sin(gamma1)**2*np.cos(delta[j1])*np.cos(delta[j2])*h[j1]*h[j2]
    
    if j1==k1 and j2!=k2:
        e=e+np.sin(gamma1)**4*np.cos(delta[j2])*np.cos(delta[k2])*h[j2]*h[k2]
    
    elif j1==k1 and j2==k2:
        
        e=e+np.sin(gamma1)**4
    else:
        
        e=e+np.sin(gamma1)**4*np.cos(delta[j1])*np.cos(delta[j2])*np.cos(delta[k1])*np.cos(delta[k2])*h[j1]*h[j2]*h[k1]*h[k2]
        
    e=e*np.sin(delta[j])*np.sin(delta[k])*h[j]*h[k]
    
    return e

def EYX(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]

    e=np.cos(gamma1)**2*np.cos(delta[j2])*h[j2]
    e+=np.cos(gamma1)**2*np.cos(delta[j1])*h[j1]
    
    if j1==k1:
        e=e-np.sin(gamma1)**2*np.cos(delta[k2])*h[k2]
    else:
        e=e-np.sin(gamma1)**2*np.cos(delta[j1])*np.cos(delta[k1])*np.cos(delta[k2])*h[j1]*h[k1]*h[k2]
    
    if j2==k2:
        e=e-np.sin(gamma1)**2*np.cos(delta[k1])*h[k1]
    else:
        e=e-np.sin(gamma1)**2*np.cos(delta[j2])*np.cos(delta[k1])*np.cos(delta[k2])*h[j2]*h[k1]*h[k2]
    
   
    e=e*np.cos(gamma1)*np.sin(gamma1)*np.sin(delta[j])*np.sin(delta[k])*h[j]*h[k]
    
    return e


def EZ(delta,gamma1,nodes,h):
    
    j=nodes[0]
    
    e=np.cos(delta[j])*h[j]
    
    return e

def EY(delta,gamma1,nodes,h):
    
    j,k,j1,j2=nodes
    
    e=np.cos(gamma1)**2*np.cos(delta[j2])*h[j2]
    
    e+=np.cos(gamma1)**2*np.cos(delta[j1])*h[j1]

    e+=np.cos(gamma1)**2*np.cos(delta[k])*h[k]
    
    e=e-np.sin(gamma1)**2*np.cos(delta[k])*np.cos(delta[j1])*np.cos(delta[j2])*h[k]*h[j1]*h[j2]

    e=e*np.sin(gamma1)*np.sin(delta[j])*h[j]
    
    return e

def EX(delta,gamma1,nodes,h):
    
    j,k,j1,j2=nodes

    e=np.cos(gamma1)**2
    
    e=e-np.sin(gamma1)**2*np.cos(delta[j1])*np.cos(delta[j2])*h[j1]*h[j2]
    
    e=e-np.sin(gamma1)**2*np.cos(delta[j1])*np.cos(delta[k])*h[j1]*h[k]
    
    e=e-np.sin(gamma1)**2*np.cos(delta[k])*np.cos(delta[j2])*h[k]*h[j2]
    
    e=e*np.cos(gamma1)*np.sin(delta[j])*h[j]
    
    return e


def energy_jk(nodes,alpha,delta,h,para):
    
    gamma1=para[0]
    beta1=para[1]
    
    j,k=nodes[:,0]
    
    nodesR=np.array([nodes[1],nodes[0]])
    
    cj=coeff_ZYX(alpha,beta1,j)
    ck=coeff_ZYX(alpha,beta1,k)
    
    
    ezz=EZZ(delta,gamma1,nodes,h)
    eyy=EYY(delta,gamma1,nodes,h)
    exx=EXX(delta,gamma1,nodes,h)
    
    ezy=EZY(delta,gamma1,nodes,h)
    eyz=EZY(delta,gamma1,nodesR,h)
    
    ezx=EZX(delta,gamma1,nodes,h)
    exz=EZX(delta,gamma1,nodesR,h)
    
    eyx=EYX(delta,gamma1,nodes,h)
    exy=EYX(delta,gamma1,nodesR,h)
    

    e=cj[0]*ck[0]*ezz+cj[1]*ck[1]*eyy+cj[2]*ck[2]*exx
    
    e=e+cj[0]*ck[1]*ezy+cj[1]*ck[0]*eyz

    e=e+cj[0]*ck[2]*ezx+cj[2]*ck[0]*exz

    e=e+cj[1]*ck[2]*eyx+cj[2]*ck[1]*exy

    return e


def energy_j(nodes,alpha,delta,h,para):
    
    gamma1=para[0]
    beta1=para[1]
    
    j=nodes[0]
    
    
    cj=coeff_ZYX(alpha,beta1,j)

    ez=EZ(delta,gamma1,nodes,h)

    ey=EY(delta,gamma1,nodes,h)
    ex=EX(delta,gamma1,nodes,h)

    
    e=cj[0]*ez+cj[1]*ey+cj[2]*ex
    

    return e
class QAOA:
    
    def __init__(self,n,qaoa_variant,pro_solve):

        self.n=n

        self.bias=qaoa_variant['solution_WS']
        
        self.edges=pro_solve

        self.variant=qaoa_variant

        self.h=np.ones(self.n)
        
        self.A=np.zeros((self.n,self.n))
         
        for i,j,w in self.edges:
             self.A[int(i)][int(j)]=w
             self.A[int(j)][int(i)]=w
    def get_expectation(self,para):

        
        if self.variant['type']=='ab':
            
            self.alpha=np.arcsin(self.bias/np.sqrt(1+self.bias**2))
            self.delta=self.alpha-np.pi/2
        
        elif self.variant['type']=='warm-start':
            epsilon_ws=self.variant['epsilon_ws']
            self.alpha=np.zeros(self.n)
            for i in range(self.n):
                
                if (1+self.bias[i])/2<=epsilon_ws:
                    self.alpha[i]=2*np.arcsin(np.sqrt(epsilon_ws))-np.pi/2
                elif (1+self.bias[i])/2>=1-epsilon_ws:
                    self.alpha[i]=2*np.arcsin(np.sqrt(1-epsilon_ws))-np.pi/2
                else:
                    self.alpha[i]=2*np.arcsin(np.sqrt((1+self.bias[i])/2))-np.pi/2
            
            self.alpha=np.pi-self.alpha
            self.delta=np.pi/2-self.alpha
            
        elif self.variant['type']=='warmest':
            self.alpha=self.bias.copy()
            self.delta=self.alpha-np.pi/2
        
        elif self.variant['type']=='standard':
            
            self.alpha=np.zeros(self.n)
            self.delta=self.alpha-np.pi/2
        
        else:
            0
        
        E=0
        for j,k,w in self.edges:
            
            nodes=node_connection(int(j),int(k), self.edges)

            E+=(1-energy_jk(nodes,self.alpha,self.delta,self.h,para))/2
            
        return -E
    
    def update_bias(self,mh,vh,alphah,para):
        beta1=0.9
        beta2=0.999
        epsilon=10**(-8)   
        
        bias_change=np.zeros(self.n)
        
        for j in range(self.n):
           
            nodes=node_connection(int(j),-1, self.edges)

            ezj=energy_j(nodes,self.alpha,self.delta,self.h,para)
            bias_change[j]=self.bias[j]+ezj/np.sqrt(3)
            
        mh=beta1*mh+(1-beta1)*bias_change
        vh=beta2*vh+(1-beta2)*bias_change**2
        
        self.bias=self.bias-alphah*mh/(np.sqrt(vh)+epsilon)

        return mh,vh 



    def get_bias_energy(self):

        e=np.ones(self.n)
 
          
        L=np.diag(np.dot(self.A,e))-self.A
         
        E=np.dot(np.sign(self.bias),np.dot(L,np.sign(self.bias)))/4

        return E


