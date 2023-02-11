Import Python3 as PYTHON;
IMPORT $;
layout:=RECORD
    DECIMAL slength;
    DECIMAL swidth;
    DECIMAL plength;
    DECIMAL pwidth;
END;

layout1:=RECORD
    STRING species;
END;

layout2:=RECORD
    set of string result;
END;
layout3:=RECORD
    set of integer final;
END;
layout4:=RECORD
    string model;
END;
STREAMED DATASET(layout2) GBCTREE(STREAMED DATASET(layout) 
independents,STREAMED DATASET(layout1) dependents,INTEGER nthreads,INTEGER ntrees,INTEGER nboost, REAL rate):=EMBED
(PYTHON : globalscope('globalscope'),persist('query'))
    import pandas as pd
    from sklearn import tree
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    import threading
    import math
    global trees,x_test,y_test
    trees=[]
    threads=[]
    independents=list(independents)
    dependents=list(dependents)
    X=pd.DataFrame(data=independents)
    Y=pd.DataFrame(data=dependents)
    x_train,x_test,y_train,y_test=train_test_split(X,Y)
    lasthread=ntrees%nthreads
    ntreesperthread=math.floor(ntrees/nthreads)
    if(lasthread!=0):
        ntreesperthread+=1
        lasthread=ntrees-(ntreesperthread*(nthreads-1))
    def gbtree(nt,lr,ne):
        for i in range(nt):
            dtree=GradientBoostingClassifier(n_estimators=ne,learning_rate=lr)
            dtree.fit(x_train,y_train)
            trees.append(dtree)
    l_rate=rate/10.0        

    for i in range(nthreads):
        if((nthreads-i)==1):
            t=threading.Thread(target=gbtree,args=(lasthread,l_rate,nboost))
            t.start()
            threads.append(t)
        else:
            t=threading.Thread(target=gbtree,args=(ntreesperthread,l_rate,nboost))
            t.start()
            threads.append(t)

    for t in range(nthreads):
        threads[t].join()
  
    return []
ENDEMBED;



STREAMED DATASET(layout3) GBRTREE(STREAMED DATASET(layout) 
independents,STREAMED DATASET(layout1) dependents,INTEGER n,INTEGER nboost, REAL rate):=EMBED
(PYTHON : globalscope('globalscope'),persist('query'),activity)
    import pandas as pd
    from sklearn import tree
    from sklearn.tree import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    import threading
    import math
    threads=[]
    independents=list(independents)
    dependents=list(dependents)
    X=pd.DataFrame(data=independents)
    Y=pd.DataFrame(data=dependents)
    x_train,x_test,y_train,y_test=train_test_split(X,Y)
    lasthread=ntrees%nthreads
    ntreesperthread=math.floor(ntrees/nthreads)
    if(lasthread!=0):
        ntreesperthread+=1
        lasthread=ntrees-(ntreesperthread*(nthreads-1))
    def gbtree(nt):
        for i in range(nt,lr,ne):
            dtree=GradientBoostingRegressor(n_estimators=ne,learning_rate=lr)
            dtree.fit(x_train,y_train)
            trees.append(dtree)             
    l_rate=rate/10.0
    for i in range(nthreads):
        if((nthreads-i)==1):
            t=threading.Thread(target=gbtree,args=(lasthread,l_rate,nboost))
            t.start()
            threads.append(t)
        else:
            t=threading.Thread(target=gbtree,args=(ntreesperthread,l_rate,nboost))
            t.start()
            threads.append(t)

    for t in range(nthreads):
        threads[t].join()
  
    return predict
ENDEMBED;

set of string CRESULT(STREAMED DATASET(layout2) 
prediction_list):=EMBED(PYTHON : globalscope('globalscope'),
persist('query'))
    import statistics
    from statistics import mode
    res=[]
    lis=list(prediction_list)
    for i in range(len(lis[0][0])):
      r=[lis[j][0][i] for j in range(len(lis))]
      r1=mode(r)
      res.append(r1)
    return res 
   
ENDEMBED;

STREAMED DATASET(layout2) CPREDICTION(STREAMED DATASET(layout2)
datas,INTEGER nthreads,INTEGER ntrees):=EMBED(PYTHON : globalscope('globalscope'),persist
('query'),activity)
    
    import threading
    import math    
    threads=[]
    global count
    count=0
    lasthread=ntrees%nthreads
    ntreesperthread=math.floor(ntrees/nthreads)
    if(lasthread!=0):
        ntreesperthread+=1
        lasthread=ntrees-(ntreesperthread*(nthreads-1))
    def gbtree(nt):
        global count
        for i in range(nt):
            dtree=trees[count]
            count+=1
            p=list(dtree.predict(x_test))
            predict.append(p)
    
    predict=[]

    for i in range(nthreads):
        if((nthreads-i)==1):
            t=threading.Thread(target=gbtree,args=(lasthread,))
            t.start()
            threads.append(t)
        else:
            t=threading.Thread(target=gbtree,args=(ntreesperthread,))
            t.start()
            threads.append(t)

    for t in range(nthreads):
        threads[t].join()
  
    return predict
ENDEMBED;

STREAMED DATASET(layout2) RPREDICTION(STREAMED DATASET(layout2)
datas,INTEGER nthreads,INTEGER ntrees):=EMBED(PYTHON : globalscope('globalscope'),persist
('query'),activity)
    
    import threading
    import math    
    threads=[]
    global count
    count=0
    lasthread=ntrees%nthreads
    ntreesperthread=math.floor(ntrees/nthreads)
    if(lasthread!=0):
        ntreesperthread+=1
        lasthread=ntrees-(ntreesperthread*(nthreads-1))
    def gbtree(nt):
        global count
        for i in range(nt):
            dtree=trees[count]
            count+=1
            p=list(dtree.predict(x_test))
            predict.append(p)
    
    predict=[]

    for i in range(nthreads):
        if((nthreads-i)==1):
            t=threading.Thread(target=gbtree,args=(lasthread,))
            t.start()
            threads.append(t)
        else:
            t=threading.Thread(target=gbtree,args=(ntreesperthread,))
            t.start()
            threads.append(t)

    for t in range(nthreads):
        threads[t].join()
  
    return predict
ENDEMBED;


set of integer RRESULT(STREAMED DATASET(layout3) 
prediction_list):=EMBED(PYTHON : globalscope('globalscope'),
persist('query'))
    import statistics
    from statistics import mean
    lis=list(prediction_list)
    res=[]
    for i in range(len(lis[0][0])):
      r=[lis[j][0][i] for j in range(len(lis))]
      r1=mean(r)
      res.append(r1)
    return res 
ENDEMBED;

STREAMED DATASET(layout4) MODEL(STREAMED DATASET(layout2)
prediction_list):=EMBED(PYTHON : globalscope('globalscope'),
persist('query'))
    from sklearn.tree import export_text
    model=[]
    for t in trees:
        for m in t.estimators_[0]:
           model.append(export_text(m))
    return model;
ENDEMBED;


predict:=GBCTREE($.independents,$.dependent,4,59,10,0.9);

OUTPUT(MODEL(predict));