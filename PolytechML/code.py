# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
import time

#Question 1:
#Télécharger les datas pour l'entrainement de l'algoritheme.

x = np.load('data/trn_img.npy')
y = np.load('data/trn_lbl.npy')

#Initialiser deux tableaux z et m.

z = list(range(10))
m = list(range(10))

#Classifier tous les images par leur étiquettes y et les stocker dans les différents éléments du tableau z.

for i in range(10):
    z[i] = x[y==i]
            
#Calculer une moyenne pour chaque classe comme un répère de la classe.
    
for i in range(10):
    m[i] = np.mean(z[i],axis=0) 

#Télécharger les datas pour tester.

t = np.load('data/dev_img.npy')
dev = np.load('data/dev_lbl.npy')

#Initialiser un tableau result pour stocker les résultats.

result = np.arange(len(t))

#Pour chaque image on calcul la distance entre lui et tous les répères des classes,
#et on prend la classe qui a la distance la moins longue comme le résultat.

for i in range(len(t)):   
    d = np.linalg.norm(t[i]-m,axis=1)
    result[i] = np.argmin(d)

#Créer un tableau error pour stocker la différence entre le résultat qu'on a obtenu et la réalité.

error = 1-(result==dev)

#On calcule le taux d'erreur.

taux = error.sum()/len(error)

########################################################

#Question 2:
#Créer une fonction pour changer facilement la dimension qu'on a besion pour tracer finalement une courbe.

def reduire(a):
    
    #Télécharger les datas pour l'entrainement de l'algoritheme.
    
    x = np.load('data/trn_img.npy')
    y = np.load('data/trn_lbl.npy')
    
    #Initialiser deux tableaux z et m.
    
    z = list(range(10))
    m = list(range(10))
    
    #On fait la réduction de la dimension par ACP et stocke les résultats dans un nouveau tableau newX
    
    pca = PCA(n_components=a)
    newX = pca.fit_transform(x)

    for i in range(10):
        z[i] = newX[y==i]
               
    for i in range(10):
        m[i] = np.mean(z[i],axis=0) 
          
    t = np.load('data/dev_img.npy')
    dev = np.load('data/dev_lbl.npy')
    newT = pca.fit_transform(t)

    result = np.arange(len(newT))

    for i in range(len(newT)):   
        d = np.linalg.norm(newT[i]-m,axis=1)
        result[i] = np.argmin(d)

    error = 1-(result==dev)
    taux = error.sum()/len(error)
    return taux

#Prendre 10 valeurs entre 0 et 30 pour tester la relation entre la fidélité des résultats et la numero de la dimension.
    
x = np.arange(0,30,3)
y = list(map(reduire,x))

#tracer la courbe.

plt.figure()
plt.plot(x,y)
plt.xlabel("dimension")
plt.ylabel("taux d'erreur")
plt.savefig("ACP_DMIN.pdf")


########################################################

#Question 3: 
#SVM


def support(a):
    
    #Pour ne pas gaspiller du temps, on prend seulement les 2000 premières images.
    
    x = np.load('data/trn_img.npy')[0:2000,:]
    y = np.load('data/trn_lbl.npy')[0:2000]
    #clf = svm.SVC(kernel='rbf', C=1, gamma=a, tol=1e-4)
    
    #Utiliser svm pour faire le classement et constater l'influence des résultats par les changements des arguments.
    
    clf = svm.SVC(kernel='poly',degree=a,gamma=1,coef0=0) 
    
    #Utiliser (x,y) pour entrainer l'algoritheme.
    
    clf.fit(x,y)
    
    test = np.load('data/dev_img.npy')[0:2000,:]
    dev = np.load('data/dev_lbl.npy')[0:2000]
    
    #Utiliser clf.predict pour savoir les prédictions par svm.
    
    result = clf.predict(test)
    
    error = 1-(result==dev)
    taux = error.sum()/len(error)
    return taux
    
x = np.arange(1,100,10)
y = list(map(support,x))
plt.figure()
plt.plot(x,y)
plt.xlabel("degree")
plt.ylabel("taux d'erreur")
plt.savefig("poly_degree.pdf")


#Enregistrer les résultats dans un fichier.npy.

'''

#Utiliser start et end pour savoir le temps d’exécution

start = time.time() 

x = np.load('data/trn_img.npy')
y = np.load('data/trn_lbl.npy')
clf = svm.SVC(kernel='poly',degree=2,gamma=1,coef0=0)
clf.fit(x,y)
test = np.load('data/tst_img.npy')
result = clf.predict(test)
np.save("test.npy",result)

end = time.time()
print("Running time: %s seconds"%(end-start))
'''

#Calculer la matrice de confusion pour ce système

'''
dev = np.load('data/dev_img.npy')
dev_true = np.load('data/dev_lbl.npy')
dev_pred = clf.predict(dev)
C=confusion_matrix(dev_true, dev_pred)
'''

########################################################

#Question 3: 
#neighbors

def neighb(a):
    
    #Pour économiser du temps, on prend seulement les 2000 premières images.
     
    x = np.load('data/trn_img.npy')[0:2000,:]
    y = np.load('data/trn_lbl.npy')[0:2000]
    
    #Utiliser neighbors pour faire le classement et constater l'influence des résultats par les changements des arguments.
     
    clfs = neighbors.KNeighborsClassifier(n_neighbors=a,algorithm = 'ball_tree', leaf_size=1)
    
    #Utiliser (x,y) pour entrainer l'algoritheme.
    
    clfs.fit(x,y)
    
    test = np.load('data/dev_img.npy')[0:2000,:]
    dev = np.load('data/dev_lbl.npy')[0:2000]
    
    #Utiliser predict pour savoir les résultats classifier par svm.
    
    result = clfs.predict(test)
    
    error = 1-(result==dev)
    taux = error.sum()/len(error)
    return taux
    
x = np.arange(1,100,10)
y = list(map(neighb,x))
plt.figure()
plt.plot(x,y)
plt.xlabel("n_neighbors")
plt.ylabel("taux d'erreur")
plt.savefig("neighbors.pdf")

