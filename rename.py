import json
import feats,files

def get_renam_fun(json_path):
    rename=read_rename(json_path)
    def helper(data_i):
        feat_i=feats.Feats()
        for name_i,rename_i in rename.items():
            print((rename_i,name_i))
            feat_i[rename_i]=data_i[name_i]
        return feat_i
    return helper

def read_rename(id):
    rename= json.load(open("%s.json" % id))
    return { files.Name(name_i):files.Name(rename_i) 
                for name_i,rename_i in rename.items()}