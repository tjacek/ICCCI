import sys
sys.path.append("..")
import numpy as np
import evol,exp,ens,learn

class Corl(object):
    def __init__(self,all_votes):
        self.all_votes=ens.Votes(all_votes)	
        self.d=[ result_i.true_one_hot() 
                  for result_i in all_votes]
    
    def __call__(self,weights):
        weights=weights/np.sum(weights)
        results=self.all_votes.results
        C=corl(results,self.d)
        n_clf=len(self.all_votes)
        loss=0
        for i in range(n_clf):
            for j in range(n_clf):
                loss+=weights[i]*weights[j] * C[i,j] 	
        return 1.0*loss

def corl(results,d):
    n_clf=len(results)
    C=np.zeros((n_clf,n_clf))
    for i in range(n_clf):
        for j in range(n_clf):
            f_i=results[i].y_pred
            f_j=results[j].y_pred
            c_ij= (f_i-d)*(f_j-d)
            C[i,j]=np.mean(c_ij)
    return C

def visualize_corl(paths,out_path):
    datasets=ens.read_dataset(paths["common"],paths["binary"]) 
    results=[learn.train_model(data_i) for data_i in datasets]
    d=[ result_i.true_one_hot() for result_i in results]
    C=corl(results,d)
#    C=np.round_(C, decimals=3) 
    np.savetxt(out_path, C)

if __name__ == "__main__":
    dataset=".."
    dir_path=None
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
    paths["common"].append("../1D_CNN/feats")
    print(paths)
    ensemble=evol.EvolEnsemble(loss=Corl)
    result=ensemble.median_exp(paths,n=10)
    result.report()
#    visualize_corl(paths,"3DHOI")