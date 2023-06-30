import numpy as np
from sklearn import datasets
from scipy.spatial import distance
import random
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabaz_score#, davies_bouldin_score 
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import time
import copy
import warnings
from scipy.spatial.distance import pdist, euclidean
from sklearn.preprocessing import MinMaxScaler
from PAM import PAM
from scipy.stats import entropy
import scipy as sp

warnings.filterwarnings("ignore")

def kl_divergence(p, q):
	return sum(p * np.log2(1.0*p/q))
	
def dpf(D):
    D2=abs(D)+1;n=len(D)
    for i in range(n):
        D2[i]=1.0*D2[i]/sum(D2[i])
    return D2
        
def Itaku(p, q):
	return sum((1.0*p/q) - np.log2(1.0*p/q))
	
def DistMat(X,P,norm,A):
    #norm:'sqeuclidean'
    n,d=X.shape
    n1,d=P.shape
    D=np.zeros((n,n1));
    if norm=='sqeuclidean':
        D = distance.cdist(X,P, 'sqeuclidean')
    elif norm=='mahalanobis':
        for i in range(n):
            for j in range(n1):
                D[i][j]=sp.spatial.distance.mahalanobis(X[i],P[j],A)
    elif norm=='itakura':
        for i in range(n):
            for j in range(n1):
                D[i][j]=entropy(X[i],qk=P[j])
    elif norm=='kullback':
        for i in range(n):
            for j in range(n1):
                D[i][j]=kl_divergence(X[i],P[j])
    return D

def Centroid(X,D,s,K,d):
    Inst=np.zeros((K,d))
    Card=np.zeros(K)
    Error = np.zeros(K)
    for k in range(K):
        ind=np.where(s==k)[0]
        Inst[k]=np.sum(X[ind],axis=0)
        Error[k] = np.sum(D[ind,k])
        Card[k]=len(ind)
    return Inst,Error,Card

def CalHar(X,K,lab,norm,A):
    n,d=X.shape
    vl=np.unique(lab)
    K=len(vl) 
    C=np.zeros((K,d))        
    for k in range(K):
        ins=np.where(lab==vl[k])[0]
        C[k]=np.mean(X[ins],axis=0)
    De3 =DistMat(X,np.array([np.mean(X,axis=0)]),norm,A)
    SUM=sum(De3)
    De4 =DistMat(X,C,norm,A)    
    SUM2=sum(np.min(De4,axis=1))    
    ch=((n-K)*(SUM[0]-SUM2))/((K-1)*SUM2)
    '''
    C=np.zeros((K,d))        
    for k in range(K):
        ins=np.where(lab==k)[0]
        C[k]=np.mean(X[ins],axis=0)
    De3 =DistMat(X,np.array([np.mean(X,axis=0)]),norm,A)
    SUM=sum(De3)
    De4 =DistMat(X,C,norm,A)    
    SUM2=sum(np.min(De4,axis=1))    
    ch=((n-K)*(SUM[0]-SUM2))/((K-1)*SUM2)
    '''
    return ch

def DavBou(X,K,lab,norm,A):
    n,d=X.shape
    vl=np.unique(lab)
    K=len(vl) 
    C=np.zeros((K,d))          
    for k in range(K):
        ins=np.where(lab==vl[k])[0]
        C[k]=np.mean(X[ins],axis=0)
        
    D = DistMat(X,C,norm,A)#distance.cdist(X,C, 'euclidean')#
    Inst0,Error0,Card0=Centroid(X,D,lab,K,d)   
    DC = DistMat(C,C,norm,A)#distance.cdist(C,C, 'euclidean')#
    B=np.zeros((K,K))
    for i in range(K-1):
        for j in range(i+1,K):
            s =(1.0/Card0[i])*Error0[i]+(1.0/Card0[j])*Error0[j]
            B[i][j]=s/DC[i][j]
            B[j][i]=s/DC[j][i]
    db=(1.0/K)*sum(np.max(B,axis=1))    
    
    '''
    C=np.zeros((K,d))          
    for k in range(K):
        ins=np.where(lab==k)[0]
        C[k]=np.mean(X[ins],axis=0)
        
    D = DistMat(X,C,norm,A)#distance.cdist(X,C, 'euclidean')#
    Inst0,Error0,Card0=Centroid(X,D,lab,K,d)   
    DC = DistMat(C,C,norm,A)#distance.cdist(C,C, 'euclidean')#
    B=np.zeros((K,K))
    for i in range(K-1):
        for j in range(i+1,K):
            s =(1/Card0[i])*Error0[i]+(1/Card0[j])*Error0[j]
            B[i][j]=s/DC[i][j]
            B[j][i]=s/DC[j][i]
    db=(1.0/K)*sum(np.max(B,axis=1)) 
    '''
    return db      
    
def MH_All(X,C,K,A,norm='sqeuclidean',delete=False,l=None,itermax=100):
    n,d=X.shape
    if l==None:
        l=1.0*n
    it=0;Err0=np.inf;eps=1.e-6;red=1.0
    L = np.sum(X, axis=0)
    BErr=np.inf;BS=[];BC=[]
    
    while (red>=eps) and (it<=itermax):
        it+=1
        D = DistMat(X,C,norm,A)#distance.cdist(X, C, 'sqeuclidean')
        S=np.argmin(D, axis=1)
        ################
        sh=-2
        DX=DistMat(X,X,norm,A)
        if len(np.unique(S))>1:
            sh=silhouette_score(DX, S,metric="precomputed")        
        ################
        Inst0,Error0,Card0=Centroid(X,D,S,K,d)
        Error1 = np.sum(D)
        Err=(l+1)*sum(Error0)-Error1
        red=(Err0-Err)/abs(Err)  
        #print('it:',it,'K:',K,'Err:',Err,'red:',red,'eps:',eps,'sh:',sh,'KMErr:',sum(Error0),'l:',l,'del:',delete)        
        if max(n-(l+1)*Card0)>=0:
            if min(Card0)==0:
                
                ind=np.where(Card0==0)
                v=np.delete(np.arange(K), ind[0])
                Err0=np.inf;red=np.inf
                
                if delete:
                    C = np.delete(C, ind[0],0)
                    K=len(v)
                    for k in range(K):
                        C[k] = (1./(n-(l+1)*Card0[v[k]]))*(L-(l+1)*Inst0[v[k]])
                else:
                    if Err<=BErr:
                        BErr=Err;BS=S;BC=C
                    C = np.delete(C, ind[0],0)
                    for k in range(K-len(v)):
                        C = D_sampling(X, C,norm,A)
            else:
                #print('----- Optimality Conditions Violated in Iteration',it,'. Try a larger value for l than',l,'------ at least',K*(max(Card0)/min(Card0))-1)
                BErr=np.inf;BS=np.zeros(n);red=np.inf;Err0=np.inf;Err=np.inf
                break#ACA PONDRIA BErr inf
        else:
            for k in range(K):
                C[k]=(1./(n-(l+1)*Card0[k]))*(L-(l+1)*Inst0[k])
            Err0 = Err
    if Err0<BErr:
        BErr=Err;BS=S;BC=C
    #print BErr
    return BC,BErr,BS,it,red,l


def MH_All_Adapt(X,C,K,A,norm='sqeuclidean',delete=False,eps0=1.0,l=None,itermax=100):
    n,d=X.shape

    it=0;Err0=np.inf;eps=1.e-6;red=1.0
    L = np.sum(X, axis=0)
    BErr=np.inf;BS=[];BC=[]
    
    D = DistMat(X,C,norm,A)#distance.cdist(X, C, 'sqeuclidean')
    S=np.argmin(D, axis=1)
    Inst0,Error0,Card0=Centroid(X,D,S,K,d)
    
    while (red>=eps) and (it<=itermax):
        it+=1
        if l==None:
            l=1.0*min(n,max(np.divide(1.0*n,Card0))+eps0)
        else:
            D = DistMat(X,C,norm,A)#distance.cdist(X, C, 'sqeuclidean')
            S=np.argmin(D, axis=1)
            Inst0,Error0,Card0=Centroid(X,D,S,K,d)       
               
        Error1 = np.sum(D)
        Err=(l+1)*sum(Error0)-Error1
        red=(Err0-Err)/abs(Err)
        
        #print('it:',it,'K:',K,'Err:',Err,'red:',red,'l:',l,'del:',delete)
        
        if max(n-(l+1)*Card0)>=0:
            if min(Card0)==0:
                ind=np.where(Card0==0)
                v=np.delete(np.arange(K), ind[0])
                Err0=np.inf;red=np.inf
                if delete:
                    C = np.delete(C, ind[0],0)
                    K=len(v)
                    for k in range(K):
                        C[k] = (1./(n-(l+1)*Card0[v[k]]))*(L-(l+1)*Inst0[v[k]])       
                else:
                    if Err<=BErr:
                        BErr=Err;BS=S;BC=C

                    C = np.delete(C, ind[0],0)
                    for k in range(K-len(v)):
                        C = D_sampling(X, C,norm,A)
            else:
                l=1.0*min(n,max(np.divide(1.0*n,Card0))+eps0)
                Err0=np.inf;red=np.inf
                BErr=np.inf;BS=[];BC=[]
        else:
            for k in range(K):
                C[k]=(1./(n-(l+1)*Card0[k]))*(L-(l+1)*Inst0[k])
            Err0 = Err
    if Err0<=BErr:
        BErr=Err;BS=S;BC=C            
    return BC,BErr,BS,it,red,l

def SECO_Adapt(X,C,K,A,l,norm='sqeuclidean',itermax=100):
    n,d=X.shape

    it=0;Err0=np.inf;eps=1.e-6;red=1.0
    L = np.sum(X, axis=0)
    BErr=np.inf;BS=[];BC=[]
    
    D = DistMat(X,C,norm,A)#distance.cdist(X, C, 'sqeuclidean')
    S=np.argmin(D, axis=1)
    Inst0,Error0,Card0=Centroid(X,D,S,K,d)
    
    while (red>=eps) and (it<=itermax):
        it+=1

        D = DistMat(X,C,norm,A)#distance.cdist(X, C, 'sqeuclidean')
        S=np.argmin(D, axis=1)
        Inst0,Error0,Card0=Centroid(X,D,S,K,d)       
               
        Error1 = np.sum(D)
        Err=(l+1)*sum(Error0)-Error1
        red=(Err0-Err)/abs(Err)
        
        #print('it:',it,'K:',K,'Err:',Err,'red:',red,'l:',l)
        
        if max(n-(l+1)*Card0)>=0:
            ind=np.argmin(Card0)
            C = np.delete(C, ind,0)
            K=K-1
            Err0=np.inf;red=np.inf
        else:
            for k in range(K):
                C[k]=(1./(n-(l+1)*Card0[k]))*(L-(l+1)*Inst0[k])
            Err0 = Err           
    return C,Err,S,it,red,l


def Runner(X,init,K,A,norm,seed):
    #init='random','D-smapling';s=0;
    np.random.seed(seed)
    DX=DistMat(X,X,norm,A)
    C=DataLoader(X,init,K,norm,A)#DataLoader(1,'D-sampling',K)
    n,d=X.shape
    
    
    ##########################################
    ########### with l fixed #################
    # l none
    ti = time.time()
    C1,BErr1,labWSC,it,rerr,l=MH_All(X,copy.deepcopy(C),K,A,norm,False)
    tf = time.time()-ti
    
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;tf=-1;ch=-1
        
    print('SECO',norm,n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l,n1,n2,(1.0*l)/n)#,davies_bouldin_score(X, labWSC),calinski_harabasz_score(X, labWSC),tf,it,l)
    
    ti = time.time()
    C1,Err,labWSC,it,rerr,l=MH_All(X,copy.deepcopy(C),K,A,norm,True)
    tf = time.time()-ti
    
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;tf=-1;ch=-1
        
    print('SECO_del',norm,n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l,n1,n2,(1.0*l)/n)#,davies_bouldin_score(X, labWSC),calinski_harabasz_score(X, labWSC),tf,it,l)

    ti = time.time()
    C1,Err,labWSC,it,red=Kmeans(X,copy.deepcopy(C),K,A,norm)
    tf = time.time()-ti
    
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;tf=-1;ch=-1
        
    print('KM',norm,n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,-1,n1,n2,-1)    
    
    #################################
    ######## Adaptativos ############
    ti = time.time()
    C1,Err,labWSC,it,rerr,l=MH_All_Adapt(X,copy.deepcopy(C),K,A,norm,False)
    tf = time.time()-ti
    
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;ch=-1;db=-1
    
    print('SECO_adp',norm,n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l,n1,n2,(1.0*l)/n)#,davies_bouldin_score(X, labWSC),calinski_harabasz_score(X, labWSC),tf,it,l)
    
    ti = time.time()
    C1,Err,labWSC,it,rerr,l=MH_All_Adapt(X,copy.deepcopy(C),K,A,norm,True)
    tf = time.time()-ti
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;ch=-1;db=-1
    print('SECO_del_adp',norm,n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l,n1,n2,(1.0*l)/n)#,davies_bouldin_score(X, labWSC),calinski_harabasz_score(X, labWSC),tf,it,l)
    #################################

def CLARA(D,n,K,norm,A,d,s,rep,DD):#(D,n,K,d,s,rep,errf,see,ini,DD,tt,l):
    #CLARA(D,n,K,d,s,rep,DD)
    err=np.infty;
    
    for r in range(rep):
        Inds=np.random.choice(n, s, replace=False)
        Ds=DD[Inds][:,Inds];Da=D[Inds]
        
        Ind=np.random.choice(s, K, replace=False)

        k_medoids = PAM(Ind,n_cluster=K, max_iter=100, tol=0.0)
        k_medoids.fit(Da.tolist())
        M=k_medoids.medoids;M=list(M);Ms=Inds[M]
        errl = np.sum(np.min(DD[:,Ms], axis=1));
        if errl < err:
            err=errl;C=D[Ms]
    DR = DistMat(D,C,norm,A);#distance.cdist(D, C, 'sqeuclidean')
    S=np.argmin(DR, axis=1)   
    return S


def DataLoader(X,initype,K,norm,A):
    
    if initype=='D-sampling':
        C = D_sampling_initialization(X, K,norm,A)
    else:
        C = X[random.sample(range(len(X)), K), :]

    return C

def Medoid(X,D,s,K,d):
    Inst=np.zeros((K,d))
    Card=np.zeros(K)
    Error = np.zeros(K)
    Cl=np.array([np.where(s==k)[0] for k in range(K)])
    for k in range(K):
        ind=Cl[k]#np.where(s==k)[0]
        #Cl[k]=ind
        Inst[k]=np.sum(X[ind],axis=0)
        Error[k] = np.sum(D[ind,k])
        Card[k]=len(ind)
    return Inst,Error,Card,Cl
    
def Cardinalities(n,s):
    K=len(np.unique(s))
    Card=np.array([len(np.where(s==k)[0]) for k in np.unique(s)])
    n1=(1.0*max(Card))/min(Card)
    n2=(1.0*n)/min(Card)
    return n1,n2

def D_sampling(X,C,norm,A):
    if len(C)==0:
        C = X[random.sample(range(len(X)), 1), :]
    else:
        D = np.min(DistMat(X,C,norm,A), axis=1)
        ind=np.random.choice(len(D), p=D/sum(D))#ind=random.choices(range(len(D)), weights=(D), k=1)
        C = np.concatenate((C, X[ind].reshape((1, C.shape[1]))))
    return C

def D_sampling_initialization(X,K,norm,A):
    C=[]
    for k in range(K):
        C=D_sampling(X,C,norm,A)
    return C

def Kmeans(X,C,K,A,norm,itermax=100):
    
    n,d=X.shape
    it=0;Err0=np.inf;eps=1.e-6;red=1.0    
    while (red>=eps) and (it<=itermax):
        it+=1
        D = DistMat(X,C,norm,A)#distance.cdist(X, C, 'sqeuclidean')
        S=np.argmin(D, axis=1)
        Inst0,Error0,Card0=Centroid(X,D,S,K,d)#Inst0,Error0,Card0,Cl=Medoid(X,D,S,K,d)
        Err=sum(Error0)
        red=(Err0-Err)/abs(Err)  
        
        #print('it:',it,'K:',K,'Err:',Err,'red:',red)
        
        for k in range(K):
            C[k]=(1./(Card0[k]))*(Inst0[k])
        Err0 = Err

    return C,Err,S,it,red
    
def mMH_All(X,C,K,A,norm,delete=False,l=None,itermax=100):
    n,d=X.shape
    if l==None:
        l=1.0*n
    it=0;Err0=np.inf;eps=1.e-6;red=1.0
    L = np.sum(X, axis=0)
    BErr=np.inf;BS=[];BC=[]
    M=copy.deepcopy(C)
    
    while (red>=eps) and (it<=itermax):
        it+=1
        D = DistMat(X,M,norm,A)#D = distance.cdist(X, M, 'sqeuclidean')
        S=np.argmin(D, axis=1)
        Inst0,Error0,Card0,Cl=Medoid(X,D,S,K,d)
        Error1 = np.sum(D)
        Err=(l+1)*sum(Error0)-Error1
        red=(Err0-Err)/abs(Err)  
        #print('it:',it,'K:',K,'Err:',Err,'red:',red,'l:',l,'del:',delete)        
        if max(n-(l+1)*Card0)>=0:
            if min(Card0)==0:
                
                ind=np.where(Card0==0)
                v=np.delete(np.arange(K), ind[0])
                Err0=np.inf;red=np.inf
                
                if delete:
                    C = np.delete(C, ind[0],0)
                    M=np.delete(M, ind[0],0)
                    Cl = np.delete(Cl, ind[0],0)
                    K=len(v)
                    for k in range(K):
                        C[k] = (1./(n-(l+1)*Card0[v[k]]))*(L-(l+1)*Inst0[v[k]])
                        Dk = DistMat(X[Cl[k]], C[k].reshape((1,d)),norm,A)#Dk = distance.cdist(X[Cl[k]], C[k].reshape((1,d)), 'sqeuclidean')
                        M[k]=X[Cl[k]][np.argmin(Dk)] 
                        
                        
                else:
                    if Err<=BErr:
                        BErr=Err;BS=S;BC=M
                    C = np.delete(C, ind[0],0)
                    M = np.delete(M, ind[0],0)
                    for k in range(K-len(v)):
                        M = D_sampling(X, M,norm,A)
            else:
                #print('----- Optimality Conditions Violated in Iteration',it,'. Try a larger value for l than',l,'------ at least',K*(max(Card0)/min(Card0))-1)
                BErr=np.inf;BS=np.zeros(n);red=np.inf;Err0=np.inf;Err=np.inf
                break#ACA PONDRIA BErr inf
        else:
            for k in range(K):
                C[k]=(1./(n-(l+1)*Card0[k]))*(L-(l+1)*Inst0[k])
                Dk = DistMat(X[Cl[k]], C[k].reshape((1,d)),norm,A)#distance.cdist(X[Cl[k]], C[k].reshape((1,d)), 'sqeuclidean')
                M[k]=X[Cl[k]][np.argmin(Dk)] 
            Err0 = Err
    if Err0<BErr:
        BErr=Err;BS=S;BC=M
    #print BErr
    return BC,BErr,BS,it,red,l


def mmMH_All_Adapt(X,C,K,A,norm,delete=False,eps0=1.0,l=None,itermax=100):
    n,d=X.shape

    it=0;Err0=np.inf;eps=1.e-6;red=1.0
    L = np.sum(X, axis=0)
    BErr=np.inf;BS=[];BC=[]
    M=copy.deepcopy(C)
        
    D = DistMat(X,M,norm,A)#D = distance.cdist(X, M, 'sqeuclidean')
    S=np.argmin(D, axis=1)
    Inst0,Error0,Card0,Cl=Medoid(X,D,S,K,d)
    
    while (red>=eps) and (it<=itermax):
        it+=1
        if l==None:
            l=1.0*min(n,max(np.divide(1.0*n,Card0))+eps0)
        else:
            D = DistMat(X,M,norm,A)#D = distance.cdist(X, M, 'sqeuclidean')
            S=np.argmin(D, axis=1)
            Inst0,Error0,Card0,Cl=Medoid(X,D,S,K,d)       
               
        Error1 = np.sum(D)
        Err=(l+1)*sum(Error0)-Error1
        red=(Err0-Err)/abs(Err)
        #print('it:',it,'K:',K,'Err:',Err,'red:',red,'l:',l,'del:',delete)           
        
        if max(n-(l+1)*Card0)>=0:
            if min(Card0)==0:
                ind=np.where(Card0==0)
                v=np.delete(np.arange(K), ind[0])
                Err0=np.inf;red=np.inf
                if delete:
                    C = np.delete(C, ind[0],0)
                    M=np.delete(M, ind[0],0)
                    Cl = np.delete(Cl, ind[0],0)
                                        
                    K=len(v)
                    for k in range(K):
                        C[k] = (1./(n-(l+1)*Card0[v[k]]))*(L-(l+1)*Inst0[v[k]])
                        Dk = DistMat(X[Cl[k]], C[k].reshape((1,d)),norm,A)#Dk = distance.cdist(X[Cl[k]], C[k].reshape((1,d)), 'sqeuclidean')
                        M[k]=X[Cl[k]][np.argmin(Dk)] 
                                                       
                else:
                    if Err<=BErr:
                        BErr=Err;BS=S;BC=M

                    C = np.delete(C, ind[0],0)
                    M = np.delete(M, ind[0],0)
                    for k in range(K-len(v)):
                        M = D_sampling(X,M,norm)
            else:
                l=1.0*min(n,max(np.divide(1.0*n,Card0))+eps0)
                Err0=np.inf;red=np.inf
                BErr=np.inf;BS=[];BC=[]
        else:
            for k in range(K):
                C[k]=(1./(n-(l+1)*Card0[k]))*(L-(l+1)*Inst0[k])
                Dk = DistMat(X[Cl[k]], C[k].reshape((1,d)),norm,A)#Dk = distance.cdist(X[Cl[k]], C[k].reshape((1,d)), 'sqeuclidean')
                M[k]=X[Cl[k]][np.argmin(Dk)] 
            Err0 = Err
    if Err0<=BErr:
        BErr=Err;BS=S;BC=M            
    return BC,BErr,BS,it,red,l


def Runner3(X,init,K,A,norm,seed):
    #init='random','D-smapling';s=0;
    np.random.seed(seed)
    DX=DistMat(X,X,norm,A)
    C=DataLoader(X,init,K,norm,A)#DataLoader(1,'D-sampling',K)
    n,d=X.shape
    
    
    ##########################################
    ########### with l fixed #################
    # l none
    ti = time.time()
    C1,BErr1,labWSC,it,rerr,l=mMH_All(X,copy.deepcopy(C),K,A,norm,False)
    tf = time.time()-ti
    
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;tf=-1;ch=-1
        
    print('SECOm',norm,n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l,n1,n2,(1.0*l)/n)#,davies_bouldin_score(X, labWSC),calinski_harabasz_score(X, labWSC),tf,it,l)
    
    ti = time.time()
    C1,Err,labWSC,it,rerr,l=mMH_All(X,copy.deepcopy(C),K,A,norm,True)
    tf = time.time()-ti
    
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;tf=-1;ch=-1
        
    print('SECOm_del',n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l,n1,n2,(1.0*l)/n)#,davies_bouldin_score(X, labWSC),calinski_harabasz_score(X, labWSC),tf,it,l)
    
 
    #################################
    ######## Adaptativos ############
    ti = time.time()
    C1,Err,labWSC,it,rerr,l=mmMH_All_Adapt(X,copy.deepcopy(C),K,A,norm,False)
    tf = time.time()-ti
    
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;ch=-1;db=-1
    
    print('SECOm_adp',n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l,n1,n2,(1.0*l)/n)#,davies_bouldin_score(X, labWSC),calinski_harabasz_score(X, labWSC),tf,it,l)
    
    ti = time.time()
    C1,Err,labWSC,it,rerr,l=mmMH_All_Adapt(X,copy.deepcopy(C),K,A,norm,True)
    tf = time.time()-ti
    if len(np.unique(labWSC))>1:
        n1,n2=Cardinalities(n,labWSC)
        ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
        ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
        db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)
    else:
        n1=-1;n2=-1;ss=-1;ch=-1;db=-1
    print('SECOm_del_adp',n,K,d,init,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l,n1,n2,(1.0*l)/n)#,davies_bouldin_score(X, labWSC),calinski_harabasz_score(X, labWSC),tf,it,l)
    #################################

def Runner4(X,K,A,norm,seed):
    init='random'
    n,d=X.shape
    DX=DistMat(X,X,norm,A)
    ti = time.time()
    DD = DistMat(X,X,norm,A)#distance.cdist(X, X, 'sqeuclidean')    
    dlabels = CLARA(X,n,K,norm,A,d,40+2*K,100,DD)
    tf = time.time()-ti
    
    if len(np.unique(dlabels))>1:
        n1,n2=Cardinalities(n,dlabels)
        ss=silhouette_score(DX, dlabels,metric="precomputed")#silhouette_score(X, dlabels)
        ch=CalHar(X,K,dlabels,norm,A)#ch=calinski_harabaz_score(X, dlabels)
        db=DavBou(X,K,dlabels,norm,A)#db=DaviesBouldin(X, dlabels)
    else:
        n1=-1;n2=-1;ss=-1;ch=-1;db=-1
    

    print('CLARA',n,K,d,init,seed,len(np.unique(dlabels)),ss,ch,db,tf,-1,-1,n1,n2,-1)#,davies_bouldin_score(X, dlabels),calinski_harabasz_score(X, dlabels),tf,-1,-1)
    
    '''
    Ind=np.random.choice(n, K, replace=False)
    t0 = time.time()    
    k_medoids = PAM(Ind,n_cluster=K, max_iter=100, tol=0.0)    
    k_medoids.fit(X.tolist())
    t1 = time.time()-t0
    M=k_medoids.medoids;M=list(M)
    D = distance.cdist(X, M, 'sqeuclidean')
    dlabels=np.argmin(D, axis=1)   
    
    if len(np.unique(dlabels))>1:
        n1,n2=Cardinalities(n,dlabels)
        ss=silhouette_score(X, dlabels)
        ch=calinski_harabaz_score(X, dlabels)
        db=DaviesBouldin(X, dlabels)
    else:
        n1=-1;n2=-1;ss=-1;ch=-1;db=-1
    

    print('PAM',n,K,d,init,seed,len(np.unique(dlabels)),ss,ch,db,tf,-1,-1,n1,n2,-1)#,davies_bouldin_score(X, dlabels),calinski_harabasz_score(X, dlabels),tf,-1,-1)
    '''    

def lw(tipo,K,text,col=0,norm='sqeuclidean'):
    #SEED=[0,1,2,3,4,5,6,7,8,9,10]
    X=np.loadtxt(text)
    if col==1:
        X=X[:,1:]
    elif col==2:
        X=X[:,::2]            
    scaler = MinMaxScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    X=np.unique(X,axis=0)   
    n,d=X.shape
    
    
    
    
    A=np.zeros((d,d))
    if norm=='mahalanobis':
        D1=D[np.random.choice(n,d, replace=False)];S=np.cov(D1);
        A=sp.linalg.inv(S);A=A/np.max(abs(A))  
    DX=DistMat(X,X,norm,A)
    T=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.10,0.25,0.50,0.75,1.00,5.00,10.00]
    
    for seed in range(1):
        np.random.seed(seed)
        C=DataLoader(X,'random',K,norm,A)#'random','D-sampling'
        
        if tipo==0:
            
            #print('##### KM ####')
            #print(C)            
            ti = time.time()
            C1,Err,labWSC,it,red=Kmeans(X,copy.deepcopy(C),K,A,norm)
            tf = time.time()-ti
            #print(C1) 
            if len(np.unique(labWSC))>1:
                ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
                ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
                db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)     
            print('KM','normal',-1,n,K,d,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,-1)

                     
        
        
        for t in T:
            lt=t*n
            if tipo==0:#no delete
                #print('##### SECO',t,lt,' ####')
                #print(C) 
                ti = time.time()
                C1,BErr1,labWSC,it,rerr,l=MH_All(X,copy.deepcopy(C),K,A,norm,False,lt)
                tf = time.time()-ti
                #print(C1) 
                #print(lt, np.unique(labWSC))
                if len(np.unique(labWSC))>1:
                    ss=silhouette_score(DX, labWSC,metric="precomputed")
                    ch=CalHar(X,K,labWSC,norm,A)
                    db=DavBou(X,K,labWSC,norm,A)          
                    print('SECO','del',t,n,K,d,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l)
                

    
            elif tipo==1:#delete
                ti = time.time()
                C1,BErr1,labWSC,it,rerr,l=MH_All(X,copy.deepcopy(C),K,A,norm,True,lt)
                tf = time.time()-ti
                
                if len(np.unique(labWSC))>1:
                    ss=silhouette_score(DX, labWSC,metric="precomputed")
                    ch=CalHar(X,K,labWSC,norm,A)
                    db=DavBou(X,K,labWSC,norm,A)          
                    print('SECO','ndel',t,n,K,d,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l)    
    return
    
        
def MainRun(text,norm='sqeuclidean',col=0):        
        
    T=5
    X=np.loadtxt(text)
    
    
    if col==1:
        X=X[:,1:]
    elif col==2:
        X=X[:,::2]        
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    X=np.unique(X,axis=0)
    
    n,d=X.shape
    A=np.zeros((d,d))
    if norm=='mahalanobis':
        D1=D[np.random.choice(n,d, replace=False)];S=np.cov(D1);
        A=sp.linalg.inv(S);A=A/np.max(abs(A))
    
    print('Method','norm','n','K','d','init','seed','Knew','Silhoutte','CalHar','DavBou','t','It','l','cmaxmin','cmin','lrel')

    for K in [3,5,10,20]:      
        for seed in range(T):
            Runner4(X,K,A,norm,seed)
            for init in ['random','D-sampling']:
                Runner3(X,init,K,A,norm,seed)
                Runner(X,init,K,A,norm,seed)




def SECOD(text,col=0,norm='sqeuclidean'):
    print('Method','tt','n','K','d','seed','Knew','Silhoutte','CalHar','DavBou','t','It','l')
    X=np.loadtxt(text)
    if col==1:
        X=X[:,1:]
    elif col==2:
        X=X[:,::2]            
    scaler = MinMaxScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    X=np.unique(X,axis=0)   
    n,d=X.shape
    A=np.zeros((d,d))
    if norm=='mahalanobis':
        D1=D[np.random.choice(n,d, replace=False)];S=np.cov(D1);
        A=sp.linalg.inv(S);A=A/np.max(abs(A))  
    DX=DistMat(X,X,norm,A)
    T=[2,3,5,10,20,50,100];
    KV=[3,5,10,20,50]
    
    for K in KV:
        for seed in range(5):
            np.random.seed(seed)
            C=DataLoader(X,'random',K,norm,A)#'random','D-sampling'
                
            ti = time.time()
            C1,Err,labWSC,it,red=Kmeans(X,copy.deepcopy(C),K,A,norm)
            tf = time.time()-ti
    
            if len(np.unique(labWSC))>1:
                ss=silhouette_score(DX, labWSC,metric="precomputed")#silhouette_score(X, labWSC)
                ch=CalHar(X,K,labWSC,norm,A)#ch=calinski_harabaz_score(X, labWSC);
                db=DavBou(X,K,labWSC,norm,A)#db=DaviesBouldin(X, labWSC)     
            print('KM',-1,n,K,d,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,-1)
    
            for t in T:
                lt=t*K
    
                ti = time.time()
                C1,BErr1,labWSC,it,rerr,l=SECO_Adapt(X,copy.deepcopy(C),K,A,lt,norm)
                tf = time.time()-ti
    
                if len(np.unique(labWSC))>1:
                    ss=silhouette_score(DX, labWSC,metric="precomputed")
                    ch=CalHar(X,K,labWSC,norm,A)
                    db=DavBou(X,K,labWSC,norm,A)          
                    print('SECO',t,n,K,d,seed,len(np.unique(labWSC)),ss,ch,db,tf,it,l)   
    return
#tipo=0;K=10
#lw(tipo,K,'trip.txt',1)
#SECOD('trip.txt',1)
#SECOD('Anuran.txt')
#SECOD('HTRU.txt')
#SECOD('turkiye.txt')
#SECOD('Tarvel.txt',1)
#SECOD('c3.txt')#Gesture
#SECOD('gt.txt')# Gas Turbine Emision
#SECOD('eb.txt')#Taminaldu
#SECOD('Relation.txt')#KEGG
#SECOD('Postures.txt')#Hand Postures
#SECOD('TIME.txt',2)#Commercial Detection

#Runner3(X,'random',K,1)
#Runner4(X,K,1)
#MainRun('3D_spatial_network.txt')
#MainRun('trip.txt',1)
#MainRun('turkiye.txt')
#MainRun('Tarvel.txt',1)
#MainRun('c3.txt')#Gesture
#MainRun('Anuran.txt')
#MainRun('HTRU.txt')
#MainRun('gt.txt')# Gas Turbine Emision
#MainRun('eb.txt')#Taminaldu
#MainRun('Relation.txt')#KEGG
#MainRun('Postures.txt')#Hand Postures
#MainRun('TIME.txt',2)#Commercial Detection
'''
X=np.loadtxt('Anuran.txt')
n,d=X.shape

scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
X=np.unique(X,axis=0)
K=3;A=0;norm='sqeuclidean'
C=DataLoader(X,'random',K,norm,A)
DX=DistMat(X,X,norm,A)
C1,BErr1,labWSC,it,rerr,l=MH_All(X,copy.deepcopy(C),K,A,norm,False)
ss=silhouette_score(X, labWSC,metric="sqeuclidean")
ss2=silhouette_score(DX, labWSC,metric="precomputed")#
ch=calinski_harabaz_score(X, labWSC);#db=DaviesBouldin(X, labWSC)
ch2=CalHar(X,K,labWSC,'sqeuclidean',A)
db2=DavBou(X,K,labWSC,'sqeuclidean',A)
'''

'''
print('Method','tipo','tt','n','K','d','seed','Knew','Silhoutte','CalHar','DavBou','t','It','l')
for K in [2,3,5,10,20]:
    for tipo in [0,1]:
        #lw(tipo,K,'Anuran.txt')
        #lw(tipo,K,'3D_spatial_network.txt')
        #lw(tipo,K,'trip.txt',1)
        #lw(tipo,K,'turkiye.txt')
        #lw(tipo,K,'Tarvel.txt',1)
        #lw(tipo,K,'c3.txt')#Gesture
        #lw(tipo,K,'HTRU.txt')
        #lw(tipo,K,'gt.txt')# Gas Turbine Emision
        #lw(tipo,K,'eb.txt')#Taminaldu
        lw(tipo,K,'Relation.txt')#KEGG
        #lw(tipo,K,'Postures.txt')#Hand Postures
        #lw(tipo,K,'TIME.txt',2)#Commercial Detection

'''