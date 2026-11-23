import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from config import FORMATIONS, PLANE_K, DENSE_SPW, DENSE_K

class FormationPlaneKNN:
    """
    K-Nearest Neighbor interpolation framework utilizing local multidimensional planes 
    to map geological structures across specified boundaries.
    """
    def __init__(self, well_ids: list[str], data_dir: any) -> None:
        """
        Initializes the plane model.
        Args:
            well_ids (list[str]): List of training well IDs.
            data_dir (Path): Directory matching data partitions.
        """
        rows=[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y']+FORMATIONS).dropna()
            except: continue
            if len(df)==0: continue
            row={'wid':wid,'x':float(df['X'].median()),'y':float(df['Y'].median())}
            for c in FORMATIONS: row[f'{c}_m']=float(df[c].median())
            rows.append(row)
        self.df=pd.DataFrame(rows); self.wmap={w:i for i,w in enumerate(self.df['wid'])}
        xy=self.df[['x','y']].to_numpy(); self.scale=np.where(xy.std(0)<1e-3,1.,xy.std(0))
        self.tree=cKDTree(xy/self.scale)
        self.xa=self.df['x'].to_numpy(); self.ya=self.df['y'].to_numpy()
        self.fa=self.df[[f'{c}_m' for c in FORMATIONS]].to_numpy()

    def impute(self, xy_q: np.ndarray, self_wid: str = None, k: int = PLANE_K) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates structurally constrained matrix heights.
        Args:
            xy_q (np.ndarray): Target coordinates.
            self_wid (str): Optional query exclusion identifier.
            k (int): Local cluster sizing.
        Returns:
            tuple[np.ndarray, np.ndarray]: Estimated matrix target values and coordinate ranges.
        """
        q=xy_q/self.scale; nf=min(k+5,len(self.df))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid in self.wmap: dist=np.where(idx==self.wmap[self_wid],np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.).astype(np.float64)
        xn=self.xa[ik]; yn=self.ya[ik]; fn=self.fa[ik]; wx=w*xn; wy=w*yn
        A=np.zeros((len(q),3,3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(w[:,:,None]*fn).sum(1)],1)
        try: coef=np.linalg.solve(A,rhs)
        except:
            coef=np.zeros((len(q),3,6))
            for r in range(len(q)):
                try: coef[r]=np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq=xy_q[:,0]; Yq=xy_q[:,1]
        pred=(Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)]=self.fa.mean(0)
        return pred,np.where(vk,dk,np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    """
    Spatial dense matrix target value interpolation framework utilizing optimized cKDTrees.
    """
    def __init__(self, well_ids: list[str], data_dir: any, spw: int = DENSE_SPW) -> None:
        """
        Initializes the spatial lookup index.
        Args:
            well_ids (list[str]): Well identifiers to index.
            data_dir (Path): Location reference path.
            spw (int): Subsampling dimension configuration constraint.
        """
        xs,ys,anccs,wids=[],[],[],[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y','ANCC']).dropna()
            except: continue
            if len(df)==0: continue
            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]
            xs.append(s['X'].values); ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values); wids.extend([wid]*len(s))
        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])
        self.ancc=np.concatenate(anccs).astype(np.float32); self.wids=np.array(wids)
        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0))
        self.tree=cKDTree(self.xy/self.scale)

    def impute(self, xy_q: np.ndarray, self_wid: str = None, k: int = DENSE_K, nfetch: int = 5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts dense target distributions for specific geographic coordinates.
        Args:
            xy_q (np.ndarray): Inquiry locations coordinates matrix.
            self_wid (str): Optional identification tracker for validation safety masks.
            k (int): Density bounds parameters.
            nfetch (int): Search window ceiling thresholds.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Metric aggregates, variations, and minimal intervals.
        """
        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale; nf=min(nfetch,len(self.ancc))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.)
        sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw); an=self.ancc[ik]
        ap=(an*w).sum(1)/safe; ap=np.where(sw<1e-9,float(self.ancc.mean()),ap)
        var=((an-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32),np.sqrt(np.maximum(var,0.)).astype(np.float32),np.where(vk,dk,np.inf).min(1).astype(np.float32)