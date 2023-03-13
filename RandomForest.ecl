Import Python3 as PYTHON;
IMPORT $;
independentss:=RECORD
    DECIMAL slength;
    DECIMAL swidth;
    DECIMAL plength;
    DECIMAL pwidth;
END;

dependents:=RECORD
    STRING species;
END;

cprediction:=RECORD
    set of string result;
END;
rprediction:=RECORD
    set of real final;
END;
treemodel:=RECORD
    string model;
END;

EXPORT RANDOMFOREST():=MODULE

    EXPORT STREAMED DATASET(cprediction) RANDCTREE(DATASET(independentss) 
    independents, DATASET(dependents) dependents,INTEGER nthreads,INTEGER ntrees):=EMBED
    (PYTHON : globalscope('globalscope'),persist('query'),activity)
        import pandas as pd
        from sklearn import tree
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.tree import export_text
        import threading
        import math
        global ctrees,x_test,y_test
        ctrees=[]
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
            ctrees.append(dtree)
                
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

    EXPORT STREAMED DATASET(rprediction) RANDRTREE():=EMBED
    (PYTHON : globalscope('globalscope'),persist('query'),activity)
        import pandas as pd
        from sklearn import tree
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import threading
        import math
        global rtrees,x_test,y_test
        rtrees=[]
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
            dtree=RandomForestRegressor(n_estimators=nt)
            dtree.fit(x_train,y_train)
            rtrees.append(dtree)
                
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

    EXPORT STREAMED DATASET(cprediction) CPREDICT(STREAMED DATASET(cprediction)
    datas,INTEGER nthreads,INTEGER ntrees):=
    EMBED(PYTHON : globalscope('globalscope'),persist('query'),activity)
        
   
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

    EXPORT STREAMED DATASET(cprediction) RPREDICT(STREAMED DATASET(rprediction)
    datas):=EMBED(PYTHON : globalscope('globalscope'),persist
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

    EXPORT set of string CRESULT(STREAMED DATASET(cprediction) 
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

    EXPORT set of real RRESULT(STREAMED DATASET(rprediction) 
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

    EXPORT STREAMED DATASET(treemodel) CMODEL(STREAMED DATASET(cprediction)
    prediction_list):=EMBED(PYTHON : globalscope('globalscope'),
    persist('query'))
        from sklearn.tree import export_text
        model=[]
        for t in ctrees:
            for m in t.estimators_:
                model.append(export_text(m))
        return model;
    ENDEMBED;

    EXPORT STREAMED DATASET(treemodel) RMODEL(STREAMED DATASET(rprediction)
    prediction_list):=EMBED(PYTHON : globalscope('globalscope'),
    persist('query'))
        from sklearn.tree import export_text
        model=[]
        for t in rtrees:
          for m in t.estimators_:
            model.append(export_text(m))
        return model;
    ENDEMBED;
    EXPORT REAL ACCURACY(set of STRING prediction_list)
    :=EMBED(PYTHON: globalscope('globalscope'),persist('query'))
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_test,prediction_list)
    ENDEMBED;
END;

