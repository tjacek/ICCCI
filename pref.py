import numpy as np
from collections import Counter
import exp
import systems,ens,learn,files

class PrefEnsemble(ens.Ensemble):
    def __init__(self,read=None,transform=None):
        super(PrefEnsemble,self).__init__(read,transform)

    def __call__(self,paths,system,binary=False,clf="LR",s_clf=None):
        votes=super(PrefEnsemble,self).__call__(paths,binary,clf,s_clf)[1]
        return voting(votes,system)

class Preferences(object):
	def __init__(self, order):
		self.order=np.array(order)

	def n_votes(self):
		return self.order.shape[0]

	def empty_votes(self):
		return np.zeros((self.n_votes(),))

	def by_vote(self):
		return [vote_i for vote_i in self.order]

	def by_order(self,as_counter=False,flip=False):
		if(as_counter):
			ordering=[Counter(ord_i) for ord_i in self.order.T]
		else:
			ordering=[ord_i for ord_i in self.order.T]
		if(flip):
			ordering.reverse()
		return ordering

	def pairwise_score(self,x,y):
		score=0
		for vote_i in self.by_vote():
			x_index=np.where(vote_i==x)[0][0]
			y_index=np.where(vote_i==y)[0][0]
			if(y_index<x_index):
				score+=1
		return score > (self.n_votes()/2)

def voting(votes,system,cf_path=None):
	if(system is None):
		system=borda_count
	y_true=votes.results[0].y_true
	names=votes.results[0].names
	votes=prepare_votes(votes)
	y_pred=[]
	for vote_i in votes:
		pref_i= to_preference(vote_i)
		y_pred.append(system(pref_i))
	return learn.Result(y_true,y_pred,names)

def prepare_votes(votes):
	y_true=votes.results[0].y_true
	n_samples=len(y_true)
	votes=np.array([ result_i.as_numpy() 
		for result_i in votes.results])
	return [votes[:,i,:] for i in range(n_samples)]

def to_preference(vote_i):
	pref=[]
	for vote_j in vote_i: 
		ord_i= np.argsort(vote_j)
		pref.append(ord_i)
	return Preferences(pref)

def exp_desc(paths,out_path=None,clf="LR",info="info",transform=None):
    ensemble=PrefEnsemble()
    all_systems=[systems.borda_count,
                 systems.bucklin,
                 systems.coombs]
    lines=[]#get_soft_voting(paths,clf,info,transform)]
    for system_i in all_systems:
        name_i=system_i.__name__
        result_i=ensemble(paths,system_i,clf=clf)
        lines.append(get_line(name_i,result_i,info))
    if(out_path):
        files.save_txt(lines,out_path)
    else:
        print(lines)

def get_line(name_i,result_i,info):
    metrics_i=exp.get_metrics(result_i)
    return "%s,%s,%s" % (name_i,info,metrics_i)

if __name__ == "__main__":
    dataset=None
    dir_path=None
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
    paths["common"].append("1D_CNN/feats")
    print(paths)
    exp_desc(paths,out_path="split_I")