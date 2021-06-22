import files

def basic_paths(dataset,dir_path,common,binary,name="dtw"):
    paths={}
    if((dataset is None) or (dir_path is None)):
        paths["binary"]=binary
        paths["common"]=files.get_paths(common,name=name)
        return paths
    paths["dir_path"]="%s/%s" % (dir_path,dataset)
    common="%s/%s" % (paths["dir_path"],common)
    paths["common"]=files.get_paths(common,name=name)
    if(binary):
        paths["binary"]="%s/%s" % (paths["dir_path"],binary)
    else:
        paths["binary"]=None
    return paths 