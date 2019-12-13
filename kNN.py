from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import gc


def change_dataset(data):
    data = data.astype('float32') / 255.0
    data=[np.array(i).reshape(3072) for i in data]
    data=np.array(data)
    return data

class KNN(object):
    def __init__(self,X,Y):
        self.Xtr = X
        self.Ytr = Y
    
    def classific(self,X,k):
        d = {}  
        for i in range(X.shape[0]):
            
            differ = np.sum(np.abs(X[i,:]-self.Xtr),axis=1)
            
            for k in range(1,k+1):
                k_neighbors = np.argpartition(differ,k)
                min_k = k_neighbors[:k]
                class_count = np.zeros(10,dtype=int)        
                for x in min_k:
                    class_count[int(self.Ytr[x])] +=1
                    
                w = np.argmax(class_count)
                
                if k not in d.keys():
                    d[k]= [w]
                elif k in d.keys():
                    
                    d[k].append(w)
            
            
        return d

k=7
n_t=100

(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = change_dataset(trainX) 
testX = change_dataset(testX) 

testX,testY=testX[:n_t],testY[:n_t]

knn = KNN(trainX,trainY)
received_test = knn.classific(testX,k)

result_classif={}

for i in received_test.keys():
    precision=0
    for j in range(len(received_test.get(i))):
        if received_test.get(i)[j] == testY[j]:
            precision=precision+1
    result_classif[i]=precision/n_t
print(result_classif)
plt.bar(range(len(result_classif)), list(result_classif.values()), align='center')
plt.xticks(range(len(result_classif)), list(result_classif.keys()))
plt.show()

gc.collect()