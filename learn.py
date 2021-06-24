import numpy as np
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import feats

class Result(object):
    def __init__(self,y_true,y_pred,names):
        if(type(y_pred)==list):
            y_pred=np.array(y_pred)
        self.y_true=y_true
        self.y_pred=y_pred
        self.names=names

    def n_cats(self):
        votes=self.as_numpy()
        return votes.shape[1]

    def as_numpy(self):
        if(self.y_pred.ndim==2):
            return self.y_pred
        else:           
            print(len(self.y_pred))
            n_cats=np.amax(self.y_true)+1
            votes=np.zeros((len(self.y_true),n_cats))
            for  i,vote_i in enumerate(self.y_pred):
                votes[i,vote_i]=1
            return votes
    
    def as_labels(self):
        if(self.y_pred.ndim==2):
            pred=np.argmax(self.y_pred,axis=1)
        else:
            pred=self.y_pred
        return self.y_true,pred

    def as_hard_votes(self):
        hard_pred=[]
        n_cats=self.n_cats()
        for y_i in self.y_pred:
            hard_i=np.zeros((n_cats,))
            hard_i[np.argmax(y_i)]=1
            hard_pred.append(hard_i)
        return np.array(hard_pred)
   
    def get_cf(self,out_path=None):
        y_true,y_pred=self.as_labels()
        cf_matrix=confusion_matrix(y_true,y_pred)
        if(out_path):
            np.savetxt(out_path,cf_matrix,delimiter=",",fmt='%.2e')
        return cf_matrix

    def true_one_hot(self):
        return to_one_hot(self.y_true,self.n_cats())

    def get_acc(self):
        y_true,y_pred=self.as_labels()
        return accuracy_score(y_true,y_pred)

    def report(self):
        y_true,y_pred=self.as_labels()
        print(classification_report(y_true, y_pred,digits=4))

    def metrics(self):
        y_true,y_pred=self.as_labels()
        return precision_recall_fscore_support(y_true,y_pred,average='weighted')

    def get_errors(self):
        errors=[]
        y_true,y_pred=self.as_labels()
        for i,y_i in enumerate(y_true):
            if(y_i!=y_pred[i]):
                errors.append( (y_i,y_pred[i],self.names[i]))
        return errors
    
    def save(self,out_path):
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)

def train_model(data,binary=False,clf_type="LR",selector=None):
    if(type(data)==str):    
        data=feats.read(data)[0]
    data.norm()
    print(data.dim())
    print(len(data))
    train,test=data.split(selector)
    model=make_model(train,clf_type)
    X_test,y_true=test.get_X(),test.get_labels()
    if(binary):
        y_pred=model.predict(X_test)
    else:
        y_pred=model.predict_proba(X_test)
    return Result(y_true,y_pred,test.names())

def make_model(train,clf_type):
    model= get_cls(clf_type)
    X_train,y_train= train.get_X(),train.get_labels()
    model.fit(X_train,y_train)
    return model

def get_cls(clf_type):
    if(clf_type=="SVC"):
        print("SVC")
        return make_SVC()
    else:
        print("LR")
        return LogisticRegression(solver='liblinear')

def make_SVC():
    params=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 50,110, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]    
    clf = GridSearchCV(SVC(C=1,probability=True),params, cv=5,scoring='accuracy')
    return clf

def to_one_hot(y,n_cats):
    one_hot=[]
    for y_i in y:
        one_hot.append(np.zeros((n_cats,)))
        one_hot[-1][y_i]=1.0
    return np.array(one_hot)


if __name__ == "__main__":
    result=train_model("1D_CNN/feats")
    result.report()