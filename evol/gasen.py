import sys
sys.path.append("..")
import numpy as np
import evol,exp,ens,learn,files

class Comb(object):
    def __init__(self,all_votes):
        self.corl=Corl(all_votes)
        self.mse=evol.MSE(all_votes)

    def __call__(self,weights):
        return self.corl(weights)+self.mse(weights) 

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
    np.savetxt(out_path, C,delimiter=',')

def gasen_exp(paths,rename_path=None,cf_path=None,n=10):
    helper=et_rename_helper(in_path)
    ensemble=evol.EvolEnsemble(loss=Comb,transform=helper)
    result,weights=ensemble.median_exp(paths,clf="LR",n=n)
    result.report()
    print(weights)
    if(cf_path):
        result.get_cf(cf_path)

def get_rename_helper(rename_path):
    import rename
    if(rename_path):
        return rename.get_renam_fun(rename_path)
    else:
        return None

def auc_exp(paths,out_path,rename_path=None,n=10):
    lines=[]
    helper=get_rename_helper(rename_path)
    for i in range(2,9):
        p_i= 0.1*(i+1)
        valid_i=evol.BaseValidation(p_i) 
        ensemble_i=evol.EvolEnsemble(valid_i,#loss=Comb,
            transform=helper)
        ensemble_i=evol.EnsExperiment(ensemble_i)
        result_i,n_clf=ensemble_i.median_exp(paths,clf="LR",n=n)
        line_i=exp.get_metrics(result_i)
        line_i="%s,%d,%s" % (str(p_i),n_clf,line_i)
        print(line_i)
        lines.append(line_i)
    files.save_txt(lines,out_path)

if __name__ == "__main__":
    dataset=".."
    dir_path=None
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
    paths["common"].append("../1D_CNN/feats")
#    visualize_corl(paths,"visualize/raw/splitII")
#    gasen_exp(paths)
    auc_exp(paths,"no_weights")#,"../rename")