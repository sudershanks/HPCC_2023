Import Python3 as PYTHON;
IMPORT $;
independents:=RECORD
    REAL field1;
    REAL field2;
    REAL field3;
    REAL field4;
    REAL field5;
    REAL field6;
    REAL field7;
    REAL field8;
    REAL field9;
    REAL field10;
    REAL field11;
    REAL field12;
    REAL field13;
    REAL field14;
    REAL field15;
    REAL field16;
    REAL field17;
    REAL field18;
    REAL field19;
    REAL field20;
    REAL field21;
    REAL field22;
    REAL field23;
    REAL field24;
    REAL field25;
    REAL field26;
    REAL field27;
END;

dependents:=RECORD
    REAL field1;
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

EXPORT DECISIONTREE():=MODULE

    SHARED cprediction:=RECORD
        set of string result;
    END;
    SHARED rprediction:=RECORD
        set of real final;
    END;
    SHARED treemodel:=RECORD
        string model;
    END;

    EXPORT STREAMED DATASET(cprediction) DECCTREE(DATASET(independents) 
    independents, DATASET(dependents) dependents,INTEGER nthreads,INTEGER ntrees):=EMBED
    (PYTHON : globalscope('globalscope'),persist('query'),activity)
        import pandas as pd
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier
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
            for i in range(nt):
                dtree=DecisionTreeClassifier()
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

    EXPORT STREAMED DATASET(rprediction) DECRTREE(DATASET(independents) 
    independents, DATASET(dependents) dependents,INTEGER nthreads,INTEGER ntrees):=EMBED
    (PYTHON : globalscope('globalscope'),persist('query'),activity)
        import pandas as pd
        from sklearn import tree
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.tree import export_text
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
            for i in range(nt):
                dtree=DecisionTreeRegressor()
                dtree.fit(x_train,y_train)
                rtrees.append(dtree)
                
        dectree(69)
    
        return []
    ENDEMBED;

    EXPORT STREAMED DATASET(cprediction) CPREDICT(STREAMED DATASET(cprediction)datas,
    INTEGER nthreads,INTEGER ntrees)
    :=EMBED(PYTHON : globalscope('globalscope'),persist('query'),activity)
        
        predict=[]
        for i in range(len(ctrees)):
            dtree=ctrees[i]
            p=list(dtree.predict(x_test))
            predict.append(p)
        return predict
    ENDEMBED;

    EXPORT STREAMED DATASET(rprediction) RPREDICT(STREAMED DATASET(rprediction)
    datas,INTEGER nthreads,INTEGER ntrees):=EMBED(PYTHON : globalscope('globalscope'),persist
    ('query'),activity)
        
        predict=[]
        for i in range(len(rtrees)):
            dtree=rtrees[i]
            p=list(dtree.predict(x_test))
            predict.append(p)
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

    EXPORt set of real RRESULT(STREAMED DATASET(rprediction) 
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
            model.append(export_text(t))
        return model;
    ENDEMBED;

    EXPORT STREAMED DATASET(treemodel) RMODEL(STREAMED DATASET(rprediction)
    prediction_list):=EMBED(PYTHON : globalscope('globalscope'),
    persist('query'))
    from sklearn.tree import export_text
    model=[]
    for t in rtrees:
        model.append(export_text(t))
    return model;
    ENDEMBED;

    EXPORT REAL ACCURACY(set of real prediction_list)
    :=EMBED(PYTHON: globalscope('globalscope'),persist('query'))
        from sklearn.metrics import r2_score
        return r2_score(y_test,prediction_list)
    ENDEMBED;
END;

