import pandas as pd
import numpy as np

def dataprocess():
    iris = pd.read_csv('./train_dir/face_feature4.csv')
    result = pd.read_csv('./labels.txt',header=None,sep=' ')
    result.columns=['smile','2','3','4']
    smile=[]
    for k in iris['Column1']:
        smile.append(result['smile'][k])
    iris['Column138']=smile
    detectable=iris['Column1']
    iris.drop(columns=['Column1'],inplace=True)
    # 处理为二分类数据
    iris['Column138'].replace(to_replace=[1,0],value=[+1,-1],inplace=True)
    return iris

#iris=dataprocess()
#print(iris.to_numpy())


#np.savetxt("finaldata.txt",iris.to_numpy(),fmt='%s')

#用自己的脸作为监测数据
def getmyself():
    myface = pd.read_csv('./train_dir/myfeatures.csv')
    myface.drop(columns=['Column1'],inplace=True)
    myface.drop(columns=['Column138'],inplace=True)
    return myface