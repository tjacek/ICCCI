import sys
sys.path.append("..")
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
import ens,exp,learn,files,feats

class EnsExperiment(object):
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def __call__(self,paths,clf="LR",n=1):
        votes,datasets=self.ensemble.make_votes(paths,clf)
        if(n==1):
            return self.single_exp(votes,datasets)
        pairs=[self.ensemble(votes,datasets) for i in range(n)]
        results,weights=zip(*pairs)
        return results,weights

    def median_exp(self,paths,clf="LR",n=1):
        results,weights=self.by_acc(paths,clf,n)
        index=(len(results)//2)
        return results[index],weights[index]

    def by_acc(self,paths,clf="LR",n=1):
        results,weights=self(paths,clf,n)
        acc=[ result_i.get_acc() for result_i in results]
        print(acc)
        indexes=np.argsort(acc)
        print(indexes)
        results=[results[i] for i in indexes]
        weights=[weights[i] for i in indexes]
        return results,weights	

class SimpleEvol(object):
    def __init__(self):	
        self.valid=BaseValidation()

    def __call__(self,votes,datasets):
        results=self.valid(datasets,"LR")
        result=ens.Votes(results).voting(False)
        return result
        
class EvolEnsemble(ens.Ensemble):
    def __init__(self,valid=None,loss=None,read=None,transform=None,
    	            full=False):
        super(EvolEnsemble,self).__init__(read,transform)
        valid,loss=init_ens(valid,loss)
        self.valid=valid
        self.loss=loss
        self.full=full

    def __call__(self,votes,datasets):
        weights,s_result=self.find_weights(datasets)
        if(self.full):
            result=votes.weighted(weights)
        else:
            result=ens.Votes(s_result).weighted(weights)
        n_clf=len(weights[weights>0.02])        
        return result,n_clf

    def find_weights(self,datasets,clf="LR"):
        results=self.valid(datasets,clf)
        loss_fun=self.loss(results)
        bound_w = [(0.01, 1.0)  for _ in results]
#        init_matrix=np.ones((6, len(bound_w)))
        result = differential_evolution(loss_fun, bound_w, 
                tol=1e-7,maxiter=3,popsize=5,polish=True)#False)
#                init=init_matrix)
        loss_fun.iter=0
        weights=result['x']
        return weights,results     

def init_ens(valid,loss):
    if(valid is None):
        valid=BaseValidation()	
    if(loss is None):
        loss=MSE
    return valid,loss

class BaseValidation(object):
    def __init__(self, p):
        self.p=p

    def __call__(self,datasets,clf="LR"):
        train,test=zip(*[data_i.split()  for data_i in datasets])
        names=list(train[0].keys())
        s_names=[ name_i for name_i in names
                    if(np.random.uniform()<self.p)]
        results=[]
        for i,train_i in enumerate(train):
            s_train_i={name_j:train_i[name_j] 
                    for name_j in s_names}
            s_data_i={**s_train_i,**test[i]}
            s_data_i= feats.Feats(s_data_i)
#            datasets.append(s_data)
            result_i=learn.train_model(s_data_i,
                binary=False,clf_type=clf)
            results.append(result_i)	
        return results

class StratValidation(object):
    def __init__(self,selector_gen=None):
        if( selector_gen is None):
            selector_gen=StratGen(test_size=0.5)	
        if(type(selector_gen)==float):
                selector_gen=StratGen(test_size=selector_gen)
        self.selector_gen=selector_gen	

    def __call__(self,datasets,clf="LR"):
        train=[data_i.split()[0] for data_i in datasets]
        names=list(train[0].keys())
        selector= self.selector_gen(names)
        results= [learn.train_model(train_i,clf_type=clf,selector=selector)
                for train_i in train]
        return results

class StratGen(object):
    def __init__(self, n_split=2,test_size=0.5):
        self.sss=StratifiedShuffleSplit(n_splits=n_split, 
                test_size=test_size, random_state=0)

    def __call__(self,names):
        self.sss.get_n_splits(names)
        y=[name_i.get_cat() for name_i in names]
        for train_index, test_index in self.sss.split(names,y):
            train_names=[ names[i] for i in train_index]
            return files.SetSelector(train_names)  


class MSE(object):
    def __init__(self,all_votes):
        self.all_votes=ens.Votes(all_votes)
        self.iter=0

    def __call__(self,weights):
        self.iter+=1
#        print(self.iter)
        weights=weights/np.sum(weights)
        result=self.all_votes.weighted(weights)
        y_true=result.true_one_hot()
        y_pred=result.y_pred
        squared_mean=[np.sum((true_i- pred_i)**2)
                for true_i,pred_i in zip(y_true,y_pred)]
        return  np.mean(squared_mean)



if __name__ == "__main__":
    dataset=".."
    dir_path=None
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
    paths["common"].append("../1D_CNN/feats")
    print(paths)
    ensemble=EvolEnsemble()
    result=ensemble(paths)
    result.report()