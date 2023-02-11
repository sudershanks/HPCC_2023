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
STREAMED DATASET(layout2) DECCTREE(STREAMED DATASET(layout) 
independents,STREAMED DATASET(layout1) dependents,INTEGER nthreads,INTEGER ntrees):=EMBED
(PYTHON : globalscope('globalscope'),persist('query'),activity)
    import pandas as pd
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import export_text
    import threading
    import math
    global ctrees,x_test,y_test
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
    def dectree(nt):
        dtree=RandomForestClassifier(n_estimators=nt)
        dtree.fit(x_train,y_train)
        trees.append(dtree)
            
    for i in range(nthreads):
        if((nthreads-i)==1):
            t=threading.Thread(target=dectree,args=(lasthread,))
            t.start()
            threads.append(t)
        else:
            t=threading.Thread(target=dectree,args=(ntreesperthread,))
            t.start()
            threads.append(t)

    for t in range(nthreads):
        threads[t].join()
  
    return []
ENDEMBED;

STREAMED DATASET(layout3) DECRTREE(STREAMED DATASET(layout) 
independents,STREAMED DATASET(layout1) dependents,INTEGER n):=EMBED
(PYTHON : globalscope('globalscope'),persist('query'),activity)
    import pandas as pd
    from sklearn import tree
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.tree import export_text
    import threading
    import math
    global rtrees,x_test,y_test
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
    def dectree(nt):
        for i in range(nt):
            dtree=DecisionTreeRegressor()
            dtree.fit(x_train,y_train)
            trees.append(dtree)
            
    for i in range(nthreads):
        if((nthreads-i)==1):
            t=threading.Thread(target=dectree,args=(lasthread,))
            t.start()
            threads.append(t)
        else:
            t=threading.Thread(target=dectree,args=(ntreesperthread,))
            t.start()
            threads.append(t)

    for t in range(nthreads):
        threads[t].join()
  
    return []
ENDEMBED;

STREAMED DATASET(layout2) CPREDICTION(STREAMED DATASET(layout2)
datas,INTEGER nthreads,INTEGER ntrees):=EMBED(PYTHON : globalscope('globalscope'),persist
('query'),activity)
    
    from sklearn.tree import DecisionTreeClassifier
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
    def dectree(nt):
        global count
        for i in range(nt):
            dtree=ctrees[count]
            count+=1
            p=list(dtree.predict(x_test))
            predict.append(p)
    
    predict=[]
    
    for i in range(nthreads):
        if((nthreads-i)==1):
            t=threading.Thread(target=dectree,args=(lasthread,))
            t.start()
            threads.append(t)
        else:
            t=threading.Thread(target=dectree,args=(ntreesperthread,))
            t.start()
            threads.append(t)

    for t in range(nthreads):
        threads[t].join()
  
    return predict
ENDEMBED;

STREAMED DATASET(layout2) RPREDICTION(STREAMED DATASET(layout3)
datas,INTEGER nthreads,INTEGER ntrees):=EMBED(PYTHON : globalscope('globalscope'),persist
('query'),activity)
    
    from sklearn.tree import DecisionTreeClassifier
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
    def dectree(nt):
        global count
        for i in range(nt):
            dtree=rtrees[count]
            count+=1
            p=list(dtree.predict(x_test))
            predict.append(p)
    
    predict=[]
    
    for i in range(nthreads):
        if((nthreads-i)==1):
            t=threading.Thread(target=dectree,args=(lasthread,))
            t.start()
            threads.append(t)
        else:
            t=threading.Thread(target=dectree,args=(ntreesperthread,))
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
        for m in t.estimators_:
           model.append(export_text(m))
    return model;
ENDEMBED;







predict:=DECCTREE($.independents,$.dependent,4,59);
OUTPUT(CPREDICTION(predict,4,59));
