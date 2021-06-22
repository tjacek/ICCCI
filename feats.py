import numpy as np
import os.path
from sklearn import preprocessing
import files

class Feats(dict):
    def __init__(self, arg=[]):
        super(Feats, self).__init__(arg)

    def dim(self):
        return list(self.values())[0].shape

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return Feats(train),Feats(test)

    def as_dataset(self):
        names=self.names()
        return self.get_X(names),self.get_labels(names),names

    def names(self):
        return sorted(self.keys(),key=files.natural_keys) 
    
    def get_X(self,names=None):
        if(names is None):
            names=self.names()
        return np.array([self[name_i] for name_i in names])

    def get_labels(self,names=None):
        if(names is None):
            names=self.names()
        return [ name_i.get_cat() for name_i in names]

    def __add__(self,feat_i):
        names=common_names(self.keys(),feat_i.keys())
        names.sort()
        new_feats=Feats()
        for name_i in names:
            x_i=np.concatenate([self[name_i],feat_i[name_i]],axis=0)
            new_feats[name_i]=x_i
        return new_feats

    def rename(self,name_dict):
        new_feats=Feats()
        for name_i,name_j in name_dict.items():
            new_feats[files.Name(name_j)]=self[name_i]
        return new_feats

    def norm(self):
        names=self.names()
        X=np.array([self[name_i] for name_i in names])
        new_X=preprocessing.scale(X)
        for i,name_i in enumerate(names):
            self[name_i]=new_X[i]

    def save(self,out_path,decimals=10):
        lines=[]
        for name_i,x_i in self.items():
            str_i=np.array2string(x_i,separator=",",precision=decimals)
            line_i="%s#%s" % (str_i.replace('\n',""),name_i)
            lines.append(line_i)
        feat_txt='\n'.join(lines)
        feat_txt=feat_txt.replace('[','').replace(']','')
        file_str = open(out_path,'w')
        file_str.write(feat_txt)
        file_str.close()

def read(in_path):
    if(type(in_path)==list):
        return [read_unified(in_path)]
    if(not os.path.isdir(in_path)):
        return [read_single(in_path)]
    return [read_single(path_i) 
                for path_i in files.top_files(in_path)]

def read_single(in_path):
    lines=open(in_path,'r').readlines()
    feat_dict={}
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)>1):
            data_i,info_i=raw[0],raw[-1]
            info_i= files.Name(info_i).clean()
            x_i=np.fromstring(data_i,sep=',')
            x_i=np.nan_to_num(x_i,nan=0.0,posinf=0.0, neginf=0.0)
            feat_dict[info_i]=x_i
    return Feats(feat_dict)

def common_names(names1,names2):
    return list(set(names1).intersection(set(names2)))