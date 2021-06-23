import sys
sys.path.append("..")
from sklearn.model_selection import StratifiedShuffleSplit
import ens,exp,learn,files

class EvolEnsemble(ens.Ensemble):
    def __init__(self,read=None,transform=None):
        super(EvolEnsemble,self).__init__(read,transform)

    def __call__(self,paths,clf="LR",s_clf=None):
        votes,datasets=super(EvolEnsemble,self).make_votes(paths,clf)
        validation_votes(datasets,clf)
#        weights=selfind_weights(datasets)
        result=votes.weighted(weights)
        return result

#    def find_weights(self,datasets,clf="LR"):
#        loss_fun=self.loss(results)
#        bound_w = [(0.01, 1.0)  for _ in results]
#        init_matrix=np.ones((6, len(bound_w)))
#        result = differential_evolution(loss_fun, bound_w, 
#                tol=1e-7,maxiter=3,popsize=5,polish=True)#False)
#                init=init_matrix)
#        loss_fun.iter=0
#        weights=result['x']
#        return weights

class StratGen(object):
    def __init__(self, n_split=2,test_size=0.5):
        self.sss=StratifiedShuffleSplit(n_splits=n_split, 
                test_size=test_size, random_state=0)

    def __call__(self,names):
        self.sss.get_n_splits(names)
        y=[name_i.get_cat() for name_i in names]
        for train_index, test_index in self.sss.split(names,y):
            train_names=[ names[i] for i in train_index]
            raise Exception(train_names)
            return files.SetSelector(train_names)

def validation_votes(datasets,clf="LR"):
    train=[data_i.split()[0] for data_i in datasets]
    names=list(train[0].keys())
    sss=StratGen()
    selector=sss(names)
    results= [learn.train_model(train_i,clf_type=clf,selector=selector)
                for train_i in train]
    return results

if __name__ == "__main__":
    dataset=".."
    dir_path=None
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
    paths["common"].append("../1D_CNN/feats")
    print(paths)
    ensemble=EvolEnsemble()
    ensemble(paths)