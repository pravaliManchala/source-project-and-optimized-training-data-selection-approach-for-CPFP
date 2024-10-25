import pandas as pd
import numpy as np
from scipy import spatial, stats
from statistics import mean
import os, math
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import metrics
from scipy import stats
#from sklearn.cluster import KMeans, SpectralClustering
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.metrics import confusion_matrix,f1_score,recall_score,roc_auc_score,matthews_corrcoef
from scipy.stats import wilcoxon
from scipy.spatial.distance import cdist
import scipy.io
import scipy.linalg

def Similarity_WPS():
    for row in range(len(all_ds)):
     te=all_ds[row]
     PATH=os.path.join(r'/content',te) #/content/EQ.csv
     Adata=pd.read_csv(PATH)
     #Adata=Adata.drop_duplicates(keep='first')
     ta=Adata.values
     ta_x=ta[:,:-1]
     for colm in range(len(all_ds)):
       tr=all_ds[colm]
       arry_te=np.zeros((4,len(Adata.columns)-1))
       arry_tr=np.zeros((4,len(Adata.columns)-1))
       if(tr==te):
        sim_score[row][colm]=1
       else:
        PATH=os.path.join(r'/content', tr)
        train=pd.read_csv(PATH)
        #train=train.drop_duplicates(keep='first')
        tra=train.values
        tar_x=tra[:,:-1]
        arry_te[0]=np.mean(ta_x,axis=0)
        arry_tr[0]=np.mean(tar_x,axis=0)
        arry_te[1]=np.median(ta_x,axis=0)
        arry_tr[1]=np.median(tar_x,axis=0)
        arry_te[2]=np.std(ta_x,axis=0)
        arry_tr[2]=np.std(tar_x,axis=0)
        arry_te[3]=np.max(ta_x,axis=0)
        arry_tr[3]=np.max(tar_x,axis=0)
        temp_cnt=0
        for i in range(4):
            difrn=arry_te[i]-arry_tr[i]
            w,p=wilcoxon(difrn)
            temp_cnt=temp_cnt + 1 if(p>0.05) else temp_cnt + 0
        sim_score[row][colm] =  temp_cnt
#all_ds = np.array(['EQ.csv','JDT.csv','ML.csv','PDE.csv'])
all_ds = np.array(['apache.csv', 'safe.csv','zxing.csv'])
#all_ds = np.array(['ant-1.7.csv','arc.csv', 'camel-1.6.csv','ivy-2.0.csv','jedit-4.2.csv','log4j-1.0.csv','lucene-2.0.csv','poi-2.0.csv','redaktor.csv','synapse-1.2.csv','tomcat.csv','velocity-1.6.csv','xalan-2.4.csv','xerces-1.3.csv'])
#all_ds=np.array(['CM1.csv','KC3.csv','MC1.csv','MC2.csv','MW1.csv','PC1.csv','PC2.csv','PC3.csv','PC4.csv','PC5.csv'])

sim_score = np.zeros((len(all_ds),len(all_ds)))
Similarity_WPS()
print(sim_score)
sim_score = (sim_score).astype(int)
row_sum = np.sum(sim_score, axis=1)
avg_scr = (row_sum-1)/(len(row_sum)-1)
print(avg_scr)
np.fill_diagonal(sim_score, 0)
binary_array = np.where(sim_score > avg_scr[:, np.newaxis], 1, 0) #(sim_score > avg_scr).astype(int)
print(binary_array)
for i in range(len(all_ds)):
  print("target data: ", all_ds[i])
  temp = np.where(binary_array[i]==1)
  print("source data: ",all_ds[temp[0]])
  print("\n")
def classi_fun3(natter,attr,train,test):
    x_tr=train[:,attr]
    y_tr=train[:,-1]
    x_te=test[:,attr]
    y_te=test[:,-1]
    sc=StandardScaler().fit(x_tr)
    x_tr=sc.transform(x_tr)
    x_te=sc.transform(x_te)
    sum=0
    for i in range(0,5):
      score=0
      classi1=KNeighborsClassifier()  #metric=ecludian/mahana, p=1/2
      classi2=LogisticRegression()
      classi3=GaussianNB()
      classi4=svm.SVC(probability=True)
      classi5=DecisionTreeClassifier()
      classi1.fit(x_tr,y_tr)
      score=roc_auc_score(y_te,classi1.predict_proba(x_te)[:,1])
      classi2.fit(x_tr,y_tr)
      score=score+roc_auc_score(y_te,classi2.predict_proba(x_te)[:,1])
      classi3.fit(x_tr,y_tr)
      score=score+roc_auc_score(y_te,classi3.predict_proba(x_te)[:,1])
      classi4.fit(x_tr,y_tr)
      score=score+roc_auc_score(y_te,classi4.predict_proba(x_te)[:,1])
      classi5.fit(x_tr,y_tr)
      score=score+roc_auc_score(y_te,classi5.predict_proba(x_te)[:,1])
      score=score/5
      sum=sum+score
      #print(score)
    sum=sum/5
    return sum
def classi_fun2(natter,attr,train,test):
    x=train[:,attr]
    y=train[:,-1]
    #sc=StandardScaler()
    #x=sc.fit_transform(x)
    classi1=KNeighborsClassifier()  #metric=ecludian/mahana, p=1/2
    classi2=LogisticRegression(max_iter=1000)
    classi3=GaussianNB()
    classi4=svm.SVC(probability=True)
    classi5=DecisionTreeClassifier()
    sum=0
    for i in range(0,4):
      score=0
      X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
      sc=StandardScaler().fit(X_train)
      X_train=sc.transform(X_train)
      X_test=sc.transform(X_test)
      classi1.fit(X_train,y_train)
      score=roc_auc_score(y_test,classi1.predict_proba(X_test)[:,1])
      classi2.fit(X_train,y_train)
      score=score+roc_auc_score(y_test,classi2.predict_proba(X_test)[:,1])
      classi3.fit(X_train,y_train)
      score=score+roc_auc_score(y_test,classi3.predict_proba(X_test)[:,1])
      classi4.fit(X_train,y_train)
      score=score+roc_auc_score(y_test,classi4.predict_proba(X_test)[:,1])
      classi5.fit(X_train,y_train)
      score=score+roc_auc_score(y_test,classi5.predict_proba(X_test)[:,1])
      score=score/5
      sum=sum+score
      #print(score)
    sum=sum/4
    return sum
def rao_algo(train,test):
    s=26  #61 #change
    t= 30 #int(0.04*len(train))
    itrn=15
    print(t)
    fit=np.empty(t)
    ffit=np.empty(itrn)
    xattr1=np.empty(s)
    xattr2=np.empty(s)
    ds=np.random.randint(2, size=(t,s))#random array with 100*21
    for i in range(0,t):
     f=int(np.sum(ds[i]))
     resf=np.where(ds[i]==1)
     fit[i]=classi_fun2(f,resf[0],train,test)
    xbin=np.argmax(fit)
    xwin=np.argmin(fit)
    xbest=fit[xbin]  #np.max(acc)
    xworst=fit[xwin]   #np.min(acc)
    xattr1=ds[xbin]
    xattr2=ds[xwin]
    for itr in range(0,itrn):
     ffit[itr]=xbest
     rk=np.random.rand(s)
     rk2=np.random.rand(s)
     for i in range(0,t):
        #tp=np.empty(s)
        cmpp=np.random.randint(t)
        while(i==cmpp):
            cmpp=np.random.randint(t)
        if(fit[cmpp]>fit[i]):
            temp=ds[xbin]-ds[xwin]
            temp=temp*rk
            temp=temp+ds[i]
            temp1=ds[cmpp]-ds[i]
            temp1=rk2*temp1
            temp=temp+temp1
            tp=np.where(temp<0.5,0,1)
        else:
            temp=ds[xbin]-ds[xwin]
            temp=temp*rk
            temp=temp+ds[i]
            temp1=ds[i]-ds[cmpp]
            temp1=rk2*temp1
            temp=temp+temp1
            tp=np.where(temp<0.5,0,1)
        f=int(np.sum(tp))
        resf=np.where(tp==1)
        tfit=classi_fun2(f,resf[0],train,test)
        if(tfit>fit[i] or (tfit==fit[i] and f<=int(np.sum(ds[i])))):
            ds[i]=tp
            fit[i]=tfit
     xbin=np.argmax(fit)
     xwin=np.argmin(fit)
     xbest=fit[xbin]  #np.max(acc)
     xworst=fit[xwin]   #np.min(acc)
     xattr1=ds[xbin]
     xattr2=ds[xwin]
     #print("final fitness: ",ffit)
     #print(xattr1,xbest)
    return xattr1,np.round(fit[xbin],3)
def clustering_IS_FS_cpdp1():
     #print("WPS,IF and FS data")
     PATH=os.path.join(r'/content', test_ds)
     test=pd.read_csv(PATH)
     XY_test=test.values
     XX_test=XY_test[:,:-1]
     inc=len(test.columns)-1
     colc=len(test.columns)
     s=1
     mask=test['bug']==s
     #mask=test['Defective']==s
     Nmin=test[mask]
     Nmaj=test[~mask]
     Xmin_y=Nmin.values
     Xmaj_y=Nmaj.values
     Xmin=Xmin_y[:,:-1]
     Xmaj=Xmaj_y[:,:-1]
     Xmin_train=np.zeros((0,colc))    #edgb
     Xmaj_train=np.zeros((0,colc))
     aXY_train=np.zeros((0,colc))
     for tr in train_set:
            PATH=os.path.join(r'/content', tr)
            train=pd.read_csv(PATH)   #len(Xmin)
            XY_train=train.values
            XX_train=XY_train[:,:-1]
            s=1
            mask=train['bug']==s
            #mask=train['Defective']==s
            Nmin=train[mask]
            Nmaj=train[~mask]
            trmin=Nmin.values
            trmaj=Nmaj.values
            trmin1=trmin[:,:-1]
            trmaj1=trmaj[:,:-1]   #np.shape(min_set)
            K=(int)(math.sqrt(len(trmin1)))
            knei = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(XX_train)
            distances, indices = knei.kneighbors(XX_test,return_distance=True)
            Tc=np.zeros((0,inc), dtype=int)
            Td=np.zeros((0,inc), dtype=int)
            for qq in range(0,len(XX_test)):  #len(np.where(XY_train[:,-1]==1)
                tmp_ones=np.sum(XY_train[indices[qq],-1])  #XY_train[58]
                tmp_zeros=K-tmp_ones
                if(tmp_ones>=tmp_zeros):
                    Td=np.vstack((Td,XX_test[qq]))
                else:
                    Tc=np.vstack((Tc,XX_test[qq]))
            print("tc,td: ",len(Tc),len(Td))
            knei1 = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(trmaj1)
            distances1, indices1 = knei1.kneighbors(Tc,return_distance=True)
            unique1, counts1 = np.unique(indices1, return_counts=True)
            sms = SMOTE()
            XX_train_temp, y_train_temp = sms.fit_resample(XY_train[:,:-1], XY_train[:,-1])
            XY_train_temp=np.column_stack((XX_train_temp,y_train_temp))
            ttp=np.where(XY_train_temp[:,-1]==1)
            trmin_temp=XY_train_temp[ttp[0],:]
            trmin1_temp=trmin_temp[:len(unique1),:]
            len_Td=len(Td)
            if(len_Td==0):
                len_Td=1
            K1=int(math.ceil(len(unique1)/len_Td))
            knei2 = NearestNeighbors(n_neighbors=K1, algorithm='auto').fit(trmin1_temp[:,:-1])
            distances2, indices2 = knei2.kneighbors(Td,return_distance=True)
            unique2, counts2 = np.unique(indices2, return_counts=True)
            avg=np.mean(counts1)
            avg2=np.mean(counts2)
            t1=np.where(counts1>=math.floor(avg))
            t11=unique1[t1]
            maj_set=trmaj[unique1] #
            t2=np.where(counts2>=math.floor(avg2))
            t22=unique2[t2]
            min_set=trmin1_temp[unique2]
            Xmin_train=np.vstack((Xmin_train,min_set))
            Xmaj_train=np.vstack((Xmaj_train,maj_set))
            #print("maj,min:  ",len(maj_set),len(min_set))
     final_train=np.vstack((Xmin_train,Xmaj_train))
     knn=np.zeros(8)
     lr=np.zeros(8)
     nb=np.zeros(8)
     svm=np.zeros(8)
     dt=np.zeros(8)
     enb=np.zeros(8)
     itr=5
     attr_list,fit=rao_algo(final_train,XY_test)
     attr_in=np.where(attr_list==1)
     print(attr_in[0])
     for qq in range(0,itr):
        arr=classification_fun2(final_train,XY_test,attr_in[0])
        knn=knn+arr[0:8]
        lr=lr+arr[8:16]
        nb=nb+arr[16:24]
        svm=svm+arr[24:32]
        dt=dt+arr[32:40]
        enb=enb+arr[40:48]
        print(np.round(arr[0:8],3), " ", np.round(arr[8:16],3), " ", np.round(arr[16:24],3), " ", np.round(arr[24:32],3), " ", np.round(arr[32:40],3)," ",np.round(arr[40:48],3))
     #print("KNN: ", np.round(knn/itr,3), "\nLR: ", np.round(lr/itr,3), "\nNB: ", np.round(nb/itr,3), "\nsvm: ", np.round(svm/itr,3), "\nDT: ", np.round(dt/itr,3),"\nensmbl: ",np.round(enb/itr,3))
     print(np.round(knn/itr,3), " ", np.round(lr/itr,3), " ", np.round(nb/itr,3), " ", np.round(svm/itr,3), " ", np.round(dt/itr,3)," ",np.round(enb/itr,3))

def classification_fun2(ftrain,ftest,attr):
    y_train=ftrain[:,-1]
    X_train=ftrain[:,attr]
    y_test=ftest[:,-1]
    X_test=ftest[:,attr]
    sc=StandardScaler().fit(X_train)
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    arr=np.zeros(48)
    cnt=0
    classi1=KNeighborsClassifier()  #metric=ecludian/mahana, p=1/2
    classi2=LogisticRegression()
    classi3=GaussianNB()
    classi4=svm.SVC(probability=True)
    classi5=DecisionTreeClassifier()
    estimators=[]
    estimators.append(('knn', classi1))
    estimators.append(('lr', classi2))
    estimators.append(('nb', classi3))
    estimators.append(('svm', classi4))
    estimators.append(('dt', classi5))
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    kfold = StratifiedKFold(n_splits=10,random_state=None,shuffle=True)
    for tr, te in kfold.split(X_train,y_train):
        classi1.fit(X_train[tr],y_train[tr])
        classi2.fit(X_train[tr],y_train[tr])
        classi3.fit(X_train[tr],y_train[tr])
        classi4.fit(X_train[tr],y_train[tr])
        classi5.fit(X_train[tr],y_train[tr])
        ensemble.fit(X_train[tr],y_train[tr])
        ypr1=classi1.predict(X_train[te])
        ypr2=classi2.predict(X_train[te])
        ypr3=classi3.predict(X_train[te])
        ypr4=classi4.predict(X_train[te])
        ypr5=classi5.predict(X_train[te])
        ypr6=ensemble.predict(X_train[te])
    y_pred1=classi1.predict(X_test)
    y_pred2=classi2.predict(X_test)
    y_pred3=classi3.predict(X_test)
    y_pred4=classi4.predict(X_test)
    y_pred5=classi5.predict(X_test)
    y_pred6=ensemble.predict(X_test)
    cm=confusion_matrix(y_test, y_pred1)
    tn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    tp=cm[1][1]
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    diff=1-FPR-TPR
    arr[cnt:cnt+8]=[FPR, TPR,f1_score(y_test,y_pred1,average="micro"), roc_auc_score(y_test, classi1.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred1),bal,diff]
    cnt=cnt+8
    cm=confusion_matrix(y_test, y_pred2)
    tn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    tp=cm[1][1]
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    diff=1-FPR-TPR
    arr[cnt:cnt+8]=[FPR, TPR, f1_score(y_test,y_pred2,average="micro"), roc_auc_score(y_test, classi2.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred2),bal,diff]
    cnt=cnt+8
    cm=confusion_matrix(y_test, y_pred3)
    tn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    tp=cm[1][1]
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    diff=1-FPR-TPR
    arr[cnt:cnt+8]=[FPR, TPR, f1_score(y_test,y_pred3,average="micro"), roc_auc_score(y_test, classi3.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred3),bal,diff]
    cnt=cnt+8
    cm=confusion_matrix(y_test, y_pred4)
    tn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    tp=cm[1][1]
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    diff=1-FPR-TPR
    arr[cnt:cnt+8]=[FPR, TPR, f1_score(y_test,y_pred4,average="micro"), roc_auc_score(y_test, classi4.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred4),bal,diff]
    cnt=cnt+8
    cm=confusion_matrix(y_test, y_pred5)
    tn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    tp=cm[1][1]
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    diff=1-FPR-TPR
    arr[cnt:cnt+8]=[FPR, TPR, f1_score(y_test,y_pred5,average="micro"), roc_auc_score(y_test, classi5.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred5),bal,diff]
    cnt=cnt+8
    cm=confusion_matrix(y_test, y_pred6)
    tn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    tp=cm[1][1]
    TPR=tp/(tp+fn)
    TNR=tn/(tn+fp)
    FPR=fp/(fp+tn)
    bal=1-(math.sqrt((0-FPR)**2+(1-TPR)**2)/math.sqrt(2))
    diff=1-FPR-TPR
    arr[cnt:cnt+8]=[FPR, TPR,f1_score(y_test,y_pred6,average="micro"), roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1]), math.sqrt(TPR*TNR), matthews_corrcoef(y_test,y_pred6),bal,diff]
    return arr

"""all_ds=np.array(['ant-1.7.csv','arc.csv', 'camel-1.6.csv','ivy-2.0.csv','jedit-4.2.csv','log4j-1.0.csv','lucene-2.0.csv','poi-2.0.csv','redaktor.csv','synapse-1.2.csv','tomcat.csv','velocity-1.6.csv','xalan-2.4.csv','xerces-1.3.csv'])
ant_set=np.array(['ivy-2.0.csv','poi-2.0.csv','tomcat.csv','xerces-1.3.csv'])
arc_set=np.array(['lucene-2.0.csv','redaktor.csv','synapse-1.2.csv'])
camel_set=np.array(['lucene-2.0.csv','redaktor.csv','tomcat.csv','velocity-1.6.csv','xerces-1.3.csv'])
ivy_set=np.array(['ant-1.7.csv','lucene-2.0.csv','xerces-1.3.csv'])
jedit_set=np.array(['ant-1.7.csv'])
log4j_set=np.array(['ant-1.7.csv'])
lucene_set=np.array(['camel-1.6.csv','ivy-2.0.csv','synapse-1.2.csv'])
poi_set=np.array(['ant-1.7.csv','camel-1.6.csv','velocity-1.6.csv','xalan-2.4.csv'])
redaktor_set=np.array(['arc.csv','lucene-2.0.csv','synapse-1.2.csv'])
synapse_set=np.array(['arc.csv','lucene-2.0.csv'])
tomcat_set=np.array(['ant-1.7.csv','camel-1.6.csv','velocity-1.6.csv','xerces-1.3.csv']) #check again
velocity_set=np.array(['camel-1.6.csv','tomcat.csv','xerces-1.3.csv'])  #camel-imp
xalan_set=np.array(['ant-1.7.csv','poi-2.0.csv','tomcat.csv'])
xerces_set=np.array(['ant-1.7.csv','camel-1.6.csv','ivy-2.0.csv','tomcat.csv','velocity-1.6.csv'])
test_ds='xalan-2.4.csv'
train_set=xalan_set """
#train_set = np.array(['ML.csv','PDE.csv']) #EQ.csv
#train_set = np.array( ['PDE.csv']) #JDT.csv
#train_set = np.array(['EQ.csv','PDE.csv']) #ML.csv
#train_set = np.array(['EQ.csv','ML.csv']) #PDE.csv
#train_set = np.array(['safe.csv','zxing.csv']) #apache.csv
#train_set = np.array(['zxing.csv']) #safe.csv
train_set = np.array(['safe.csv']) #zxing.csv
#clustering_IS_FS_cpdp1()
#all_ds = np.array(['ant-1.7.csv','arc.csv', 'camel-1.6.csv','ivy-2.0.csv','jedit-4.2.csv','log4j-1.0.csv','lucene-2.0.csv','poi-2.0.csv','redaktor.csv','synapse-1.2.csv','tomcat.csv','velocity-1.6.csv','xalan-2.4.csv','xerces-1.3.csv'])
#all_ds = np.array(['CM1.csv','KC3.csv','MC1.csv','MC2.csv','MW1.csv','PC1.csv','PC2.csv','PC3.csv','PC4.csv','PC5.csv'])
#all_ds= np.array(['EQ.csv','JDT.csv','ML.csv','PDE.csv'])
all_ds = np.array(['apache.csv', 'safe.csv','zxing.csv'])
test_ds = all_ds[2]
"""PATH=os.path.join(r'/content', test_ds)
train=pd.read_csv(PATH)"""
clustering_IS_FS_cpdp1()
"""
for i in range(len(all_ds)):
  test_ds = all_ds[i]
  print("target data: ", test_ds)
  temp = np.where(binary_array[i]==1)
  train_set = all_ds[temp[0]]
  print("source data: ", train_set)
  print("\n")
  clustering_IS_FS_cpdp1()"""