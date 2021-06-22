import numpy as np

def borda_count(prefer):
	votes=prefer.empty_votes()
	for prefer_i in prefer.by_vote():
		for j,cat_j in enumerate(prefer_i):
			votes[cat_j]+=j
	return np.argmax(votes)

def bucklin(prefer):
	votes=prefer.empty_votes()
	half=votes.shape[0]/2
	for count_i in prefer.by_order(True,True):
		for cat,n in count_i.items():
			votes[cat]+=n
			if(np.amax(votes)>=half):
				return np.argmax(votes)

def k_aproval(prefer,k=2):
	aprov=prefer.order[:,-k:].flatten()
	votes=Counter(aprov)
	return votes.most_common()[0][0]

def coombs(prefer):
	counters=prefer.by_order(as_counter=True,flip=True)
	first=counters[0]
	last=counters.pop()
	while(first!=last):
		if(sum(last.values())==0):
			last=counters.pop()
		best=first.most_common()[0]
		major=sum(first.values())/2
		if(major<best[1]):
			return best[0]
		worst= last.most_common()[0]
		first[worst[0]]=0
		last[worst[0]]=0
	raise Exception(first)

def all_pairs(n_votes):
	pairs=[]
	for i in range(n_votes):
		for j in range(i,n_votes):
			pairs.append((i,j))
	return pairs

def pair_win(pair,pref_i):
    return (pref_i==pair[0])<(pref_i==pair[1])