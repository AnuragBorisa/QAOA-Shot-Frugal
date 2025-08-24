from __future__ import annotations
import numpy as np
from typing import Callable,Dict,List,Tuple,Any

#type: callable(params)->(value,aux)
Objective = Callable[[np.ndarray],Tuple[float,Dict]]
#type: callable(theta_plus,theta_minus,k,**kwargs)->(fp,fn,auxp,auxn)
ObjectivePair = Callable[[np.ndarray,np.ndarray,int],Tuple[float,float,Dict,Dict]]

class SPSA:

    def __init__(self,a:float=0.1,c:float=0.1,alpha:float=0.602,
                 gamma:float=0.101,A:float=10.0,seed: int|None = None):
        self.a0 ,self.c0 , self.alpha,self.gamma , self.A = a ,c , alpha,gamma,A
        self.rng = np.random.default_rng(seed)

    def ak(self,k:int)->float:
        return self.a0 / ((k+1+self.A) ** self.alpha)
    
    def ck(self,k:int)->float:
        return self.c0 / ((k+1)**self.gamma)
    
    def step(self,obj:Objective,params:np.ndarray,k:int)->Tuple[np.ndarray,Dict]:
        d = len(params)
        Delta = self.rng.choice([-1.0,1.0] ,size=d)
        ck = self.ck(k)
        thetap = params + ck * Delta
        thetam = params - ck * Delta

        fp , auxp = obj(thetap)
        fm , auxm = obj(thetam)

        ghat = (fp-fm)/(2.0 * ck) * (1.0 / Delta)
        ak = self.ak(k)
        new_params = params - ak * ghat

        aux = {"fp": fp, "fm": fm, "ghat": ghat, "ak": ak, "ck": ck, "auxp": auxp, "auxm": auxm}

        return new_params,aux
    
    def minimise(self , obj:Objective,x0:np.ndarray,maxitter:int = 200)->Tuple[np.ndarray,Dict]:
        params = x0.copy()
        history = {"value":[],"params":[],"aux":[]}
        best_val =  float('inf')
        best_params = params.copy()

        for k in range(maxitter):
            params , aux = self.step(obj,params,k)
            val,_ = obj(params)
            history["value"].append(val)
            history["params"].append(params)
            history["aux"].append(aux)
            if val < best_val:
                best_val ,best_params = val , params.copy()
            
        return best_params,{"best_val":best_val,"history":history}
    
def default_shots_schedule(k:int ,base:int=256,growth:float=0.5,cap:int=4096)->int:
        return int(min(cap,base*((1+k) ** growth)))
    
def default_seed_schedule(k:int,base_seed:int=12345)->int:
        return int(base_seed+k)
    

class ShotFrugalSPSA(SPSA):

    def step_pair(self,obj_pair:ObjectivePair,params:np.ndarray,k:int,**obj_kwargs:Any)->Tuple[np.ndarray,Dict]:
        d = len(params)
        Delta = self.rng.choice([-1.0,1.0],size=d)
        ck = self.ck(k)
        thetap = params + ck * Delta
        thetam = params - ck * Delta

        fp,fm,auxp,auxm = obj_pair(thetap,thetam,k,**obj_kwargs)

        ghat = (fp-fm)/(2.0 * ck) *(1/Delta)

        ak = self.ak(k)
        new_params = params - ak*ghat

        aux = {
            "fp": fp, "fm": fm, "ghat": ghat, "ak": ak, "ck": ck,
            "auxp": auxp, "auxm": auxm, "Delta": Delta
        }

        return new_params , aux 
    
    def minimise_pair(self,obj_pair:ObjectivePair,x0:np.ndarray,maxiter:int=200,**obj_kwargs:Any)->Tuple[np.ndarray,Dict]:
        params = x0.copy()
        history = {"value":[],"params":[],"aux":[]}
        best_val = float('inf')
        best_params = params.copy()

        

        for k in range(maxiter):
            params , aux = self.step_pair(obj_pair,params,k,**obj_kwargs)
            eval_current = obj_kwargs.get("eval_current")
            
            if eval_current is None:
                val = aux['fp']
            else:
                val,_ = eval_current(params,k)
            history["value"].append(val)
            history['params'].append(params)
            history["aux"].append(aux)

            if val < best_val :
                    best_val , best_params = val , params.copy()

        return best_params , {"best_val":best_val,"history":history}
    