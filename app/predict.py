#!./venv/bin/python
# -*- coding: utf-8 -*-
"""
----------------------------------------------------------
UNIVERSIDAD DE SEVILLA
ESCUELA TÉCNICA SUPERIOR DE INGENIERÍA INFORMÁTICA
GRADO INGENIERÍA INFORMÁTICA - TECNOLOGÍAS INFORMÁTICAS

Trabajo fin de grado: Predicción de la alergenicidad potencial
de las proteínas alimentariasmediante diferentes técnicas de
Aprendizaje Automático.

Autor: Francisco Manuel García Moreno
Tutor: Miguel Ángel Gutiérrez Naranjo
Fecha: 30-06-2016 (ultima modificacion: 27/04/2017)
----------------------------------------------------------
"""

import os
import math
from collections import Counter
import sys
import numpy as np
import pydotplus
# from sklearn.externals.six import StringIO
from six import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

try:
    from .alignment import create_alignments_files
    from .preprocessing import extract_all_features_and_classifications
    from .preprocessing import features_for_extracting_to_string
except SystemError:
    from alignment import create_alignments_files
    from preprocessing import extract_all_features_and_classifications
    from preprocessing import features_for_extracting_to_string

def tuning_model(method, score='precision', featToExtract=[True, True], m=1, kfolds=10, posAlFile="a_cdep.txt"
                 , negAlFile="a_cden.txt", testAlFile="", reduction=0, verbose=False):
    if verbose:
        print("Extrayendo features..")
    f, c, ft, ct, p, pt = extract_all_features_and_classifications(featToExtract, m, posAlFile, negAlFile, testAlFile)
    if reduction > 0:
        #misma proporción de alérgenos que de no alérgenos al reducir los conjuntos
        lastClass='non-allergen'
        lastTestClass = 'non-allergen'
        fa, ca, fta, cta = [], [], [], []
        for i in range(reduction):
            for j,cl in enumerate(c):
                if cl != lastClass:
                    fa.append(f[j])
                    ca.append(cl)
                    lastClass = 'allergen' if lastClass == 'non-allergen' else 'non-allergen'
                    break
            for j,cl in enumerate(ct):
                if len(ct)>0 and cl != lastTestClass:
                    fta.append(ft[j])
                    cta.append(cl)
                    lastTestClass = 'allergen' if lastTestClass == 'non-allergen' else 'non-allergen'
                    break
        f, ft, c, ct = fa, fta, ca, cta
    X_train, X_test, y_train, y_test = f, ft, c, ct

    # dividimos el dataset en dos partes iguales
    if testAlFile == "":
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=0)

    if verbose:
        print("# Tuning hyper-parameters for %s" % score)
        print()

    if method=='knn':
        ks = list(filter(lambda x: x % 2 != 0, range(1, max(min(30, len(X_train)-5), 1)))) # k impar
        tuned_parameters = [{'n_neighbors': ks, 'metric': ['euclidean', 'minkowski']
                                , 'weights': ['distance', 'uniform']
                                , "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
                                , "leaf_size":[1, 15, 30, 50, 75, 100], "p":[1,2,3]}]
    elif method=="dt":
        tuned_parameters = [{"min_samples_leaf":[1, 2, 3, 5, 8, 10, 15, 30, 50, 100]
                                , "criterion":['gini', 'entropy'], "max_depth":[5, 10, 30, 50]}]

    elif method=="nb":
        tuned_parameters = [{"priors": [None, [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]}]
    elif method=="mlp":
        tuned_parameters = [{"hidden_layer_sizes": [(100,), (150,), (200,), (300,), (500,)]
                                ,"max_iter": [200, 300, 500], "solver": ['adam', 'sgd', 'lbfgs']
                                , "activation": ['identity', 'logistic', 'tanh', 'relu']}]
    elif method=="rbm":
        tuned_parameters = [{"rbm__learning_rate": [0.1, 0.01, 0.001],"rbm__n_iter": [20, 40, 80, 100]
                                ,"rbm__n_components": [50, 100, 200, 300, 500, 1000]
                                , "dt__min_samples_leaf": [1, 2, 3, 5, 8, 10, 15, 30, 50, 100]
                                , "dt__criterion":['gini', 'entropy'], "dt__max_depth":[5, 10, 30, 50]
                             }]
            # , "logistic__C": [1.0, 10.0, 100.0, 500.0, 1000.0]}]
    elif method=="km":
        tuned_parameters = [{"n_clusters":[2, 3, 4], "max_iter": [300, 500]
                                , "algorithm": ['auto', 'full' or 'elkan']}]

    model = create_prediction_model(method)
    if score != None:
        clf = GridSearchCV(model, tuned_parameters, cv=kfolds, scoring='%s_macro' % score)
    else:
        clf = GridSearchCV(model, tuned_parameters, cv=kfolds)

    if method != "km":
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train)

    if verbose:
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    return clf.best_params_, f, c, ft, ct, p, pt



def create_prediction_model(method, params={}):
    '''
    Crea el modelo de predicción
    :param method: la técnica de aprendizaje que se desea utilizar
    :param params: parámetros adicionales de configuración del modelo
    :return: Devuelve el modelo de predicción configurado, con los parámetros indicados
    '''
    model = None
    # eleccion del metodo de clasificación
    if method == "knn": # el modelo kNN
        distance = 'euclidean'
        k = 3
        weights='distance'
        alg='auto'
        le=1
        p=1
        param = params.get("n_neighbors")
        if param != None:
            k = param
        param = params.get("metric")
        if param != None and param != "":
            distance = param
        param = params.get("weights")
        if param != None and param != "":
            weights = param
        param = params.get("p")
        if param != None and param != "":
            p = param
        param = params.get("algorithm")
        if param != None and param != "":
            alg = param
        param = params.get("leaf_size")
        if param != None and param != "":
            le = param
        model = KNeighborsClassifier(n_neighbors=k, metric=distance, weights=weights, p=p, algorithm=alg, leaf_size=le)
    elif method == "dt":
        msl=1
        c='gini'
        md=None
        param = params.get("min_samples_leaf")
        if param != None:
            msl = param
        param = params.get("criterion")
        if param != None:
            c = param
        param = params.get("max_depth")
        if param != None:
            md = param
        model = tree.DecisionTreeClassifier(min_samples_leaf=msl, criterion=c, max_depth=md)
    elif method == "nb":
        p=None
        param = params.get("priors")
        if param != None:
            p = param
        model = GaussianNB(priors=p)
    elif method == "rf":
        # http://stackoverflow.com/questions/20463281/how-do-i-solve-overfitting-in-random-forest-of-python-sklearn
        model = RandomForestClassifier(n_estimators=20, max_features=0.4)
    elif method == "mlp":
        h=(100,)
        mi=200
        s='adam'
        ac='relu'
        param = params.get("hidden_layer_sizes")
        if param != None:
            h = param
        param = params.get("max_iter")
        if param != None:
            mi = param
        param = params.get("solver")
        if param != None:
            s = param
        param = params.get("activation")
        if param != None:
            ac = param
        model = MLPClassifier(hidden_layer_sizes=h, max_iter=mi, solver=s, activation=ac)
    elif method == "svm":
        model = svm.SVC()
    elif method == "log":
        model = linear_model.LogisticRegression(C=100.0)
    elif method == "rbm":
        lr=0.1
        ni=20
        nc=100
        mod="dt"
        modpar={'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 3}
        param = params.get("model")
        if param != None:
            mod=param
        param = params.get("mod_par")
        if param != None:
            modpar = param
        param = params.get("rbm__learning_rate")
        if param != None:
            lr = param
        param = params.get("rbm__n_iter")
        if param != None:
            ni = param
        param = params.get("rbm__n_components")
        if param != None:
            nc = param

        model = create_prediction_model(method=mod, params=modpar)
        rbm = BernoulliRBM(learning_rate=lr, n_components=nc, n_iter=ni, random_state=0)

        # finalmente, el clasificador RBM-MODEL (MODEL: knn, dt, nb, mlp, etc.)
        model = Pipeline([("rbm", rbm), (mod, model)])
    elif method=="km":
        c = 2
        ni=300
        a='auto'
        param = params.get("n_clusters")
        if param != None:
            c = param
        param = params.get("max_iter")
        if param != None:
            ni = param
        param = params.get("algorithm")
        if param != None:
            a = param
        model = KMeans(n_clusters=c, max_iter=ni, algorithm=a)

    return model

def perform_prediction(method, features, classifications, test_data
    , tests_classif, kFolds=0, params={}, printNativeClassReport=False, plotModel=False):
    '''
    Entrada:
    :param method: la técnica de Aprendizaje Automático que se va a utilizar. Los valores aceptados son:
    "knn" (k Nearest Neighbors), "dt" (Decision Tree), "nb" (Naïve Bayes), "rf" (Random forest), "mlp" (Multilayer perceptron)
    :param features: una lista de las características extraídas
    :param classifications: una lista con las clasificaciones de estas características
    :param test_data: una lista con las características de los datos de prueba
    :param tests_classif: una lista con las clasificaciones de los datos de prueba
    :param kFolds: total de folds para dividir el conjunto de entrenamiento y
    realizar la validación cruzada. 0 o menor para no realizar validación,
    1 para realizar validación Leave-one-out y mayor que 1 para CV normal.
    :param params: Diccionario cuyas claves representan parámetros específicos de
    cada método de Aprendizaje Automático. A continuación se detallan los
    parámetros aceptados por cada método:

    Parámetros para kNN: "k":integer (el número de vecinos)
        , "d":"string" (distancia: consultar valores de parámetro 'metric' de
        la clase KNeighborsClassifier)

    :param printNativeClassReport: True para imprimir el informe de la clasificacion que
    proporciona scikitlearn.

    :param plotModel: si es True imprime una figura del modelo
    
    Salida:
    Devuelve tupla con 11 elementos:
    1) las train features, 2) train classfications, 3) test features
    , 4) test classifications
    5) la predicción, en porcentaje, realizada de la clasificación de cada uno de los datos de prueba;
    6) la precision, en porcentaje, de las predicciones correctamente clasificadas;
    7) la sensibilidad, en porcentaje, de las predicciones (positivas) de la clase "allergen" correctamente clasificadas;
    8) la especificidad, en porcentaje, de las predicciones (negativas) de la clase "non-allergen" correctamente clasificadas;
    9) el valor de PPV, en porcentaje
    10) el score F1, en porcentaje
    11) el valor de MCC, en porcentaje

    :return Tupla: (featsTrain,classTrain,featsTest,classTest,class_predicted,accuracy,
    sensitivity,specifity,ppv,f1Score,mcc)
     '''
    featsTrain=[]
    classTrain=[]
    featsTest=[]
    classTest=[]
    class_predicted=[]
    accuracy=0
    sensitivity=0
    specifity=0
    ppv=0
    f1Score=0
    mcc=0
    measures=[] #medidas parciales


    if method == "rbm":
        features = np.asarray(features, 'float32')
        features = scale(features)  # 0-1 scaling

        if len(test_data) > 0:
            test_data = np.asarray(test_data, 'float32')
            test_data = scale(test_data)  # 0-1 scaling

    #validacion cruzada
    if kFolds > 1:
        skf = StratifiedKFold(n_splits=kFolds, shuffle=True)
        #i=0
        if method != "rbm":
            features = np.array(features)
        classifications = np.array(classifications)
        for train_index, test_index in skf.split(features,classifications):
            featsTrain, featsTest = features[train_index], features[test_index]
            classTrain, classTest = classifications[train_index], classifications[test_index]

            # construcción del modelo de predicción
            model = create_prediction_model(method, params)
            if method=="km":
                model.fit(featsTrain)
            else:
                model.fit(featsTrain, classTrain)
            class_predicted=model.predict(featsTest)

            if method=="km":
                class_predicted = np.array(["allergen" if i==0 else "non-allergen" for i in class_predicted])

            accuracy,sensitivity,specifity,ppv,f1Score,mcc=classification_performance(classTest, class_predicted)#model.score(X_test, y_test)
            measures.append([accuracy,sensitivity,specifity,ppv,f1Score,mcc])

            if printNativeClassReport and method!="km":
                print(classification_report(classTest, class_predicted, target_names=["allergen", "non-allergen"]))
            elif printNativeClassReport and method=="km":
                classTest = np.array([0 if i == "allergen" else 1 for i in classTest])
                class_predicted = np.array([0 if i == "allergen" else 1 for i in class_predicted])
                print(classification_report(classTest, class_predicted, target_names=["allergen", "non-allergen"]))
                print(confusion_matrix(classTest, class_predicted))

        accuracy=sum(a for a,s,sp,ppv,fs,mcc in measures)/len(measures)
        sensitivity=sum(s for a,s,sp,ppv,fs,mcc in measures)/len(measures)
        specifity=sum(sp for a,s,sp,ppv,fs,mcc in measures)/len(measures)
        ppv=sum(ppv for a,s,sp,ppv,fs,mcc in measures)/len(measures)
        f1Score=sum(fs for a,s,sp,ppv,fs,mcc in measures)/len(measures)
        mcc=sum(mcc for a,s,sp,ppv,fs,mcc in measures)/len(measures)

    #validacion leave-one-out
    elif kFolds == 1:
        features=np.array(features)
        classifications=np.array(classifications)
        loo = LeaveOneOut()

        #creamos listas de las clasificaciones de los datos tomados para test y la prediccion
        for train, test in loo.split(features):
            feat_train, feat_test = features[train], features[test]
            class_train, class_test = classifications[train], classifications[test]
            class_test = np.array([0 if i=="allergen" else 1 for i in class_test])
            featsTrain.extend(feat_train)
            classTrain.extend(class_train)
            featsTest.append(feat_test[0])
            classTest.append(class_test[0])
            model = create_prediction_model(method, params)
            if method != "km":
                model.fit(feat_train, class_train)
            else:
                model.fit(feat_train)
            cpred=model.predict(feat_test)
            class_predicted.append(cpred)
            if method == "km":
                class_predicted = np.array(["allergen" if i == 0 else "non-allergen" for i in class_predicted])
            accuracy, sensitivity, specifity, ppv, f1Score, mcc = classification_performance(classTest,
                                                                                             class_predicted)
            measures.append([accuracy, sensitivity, specifity, ppv, f1Score, mcc])

        accuracy = sum(a for a, s, sp, ppv, fs, mcc in measures) / len(measures)
        sensitivity = sum(s for a, s, sp, ppv, fs, mcc in measures) / len(measures)
        specifity = sum(sp for a, s, sp, ppv, fs, mcc in measures) / len(measures)
        ppv = sum(ppv for a, s, sp, ppv, fs, mcc in measures) / len(measures)
        f1Score = sum(fs for a, s, sp, ppv, fs, mcc in measures) / len(measures)
        mcc = sum(mcc for a, s, sp, ppv, fs, mcc in measures) / len(measures)

    #sin validación cruzada, test completo del conjunto de prueba dado
    if kFolds == 0 and len(test_data) > 0:
        featsTrain=features
        classTrain=classifications
        featsTest=test_data
        model = create_prediction_model(method, params)
        if method != "km":
            model.fit(featsTrain, classTrain)
        else:
            model.fit(featsTrain)
        class_predicted=model.predict(featsTest)
        if plotModel and method=='dt':
            plot_decision_tree(model)
        
    return featsTrain,classTrain,featsTest,classTest,class_predicted,accuracy,sensitivity,specifity,ppv,f1Score,mcc


def scale(X, eps = 0.001):
    '''
    Source: http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/

    Scale the data points s.t the columns of the feature space
    (i.e the predictors) are within the range [0, 1].
    The scale function takes two parameters, our data matrix X and an epsilon
    value used to prevent division by zero errors.
    For each of the 784 columns in the matrix, we subtract the value from the
    minimum of the column and divide by the maximum of the column. By doing this,
    we have ensured that the values of each column fall into the range [0, 1].
    '''
    return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)


def totalElementsOfClass(classifications, classType="allergen"):
    '''
    Entrada:
    :param classifications: una lista con todas las clasificaciones
    :param classType: el tipo de la clasificacion buscada

    Salida:
    :return Devuelve el total de clasificaciones encontradas de la clase indicada.
    '''
    return sum(1 for c in classifications if c==classType)


def classification_performance(test_classif, predicted_class):
    '''
    Entrada:
    :param test_classif: las clasificaciones del conjunto de test conocidas
    :param predicted_class: las clasificaciones predecidas

    Salida:
    :return Tupla:
    1) la predicción, en porcentaje, realizada de la clasificación de cada uno de los datos de prueba;
    2) la precision, en porcentaje, de las predicciones correctamente clasificadas;
    3) la sensibilidad, en porcentaje, de las predicciones (positivas) de la clase "allergen" correctamente clasificadas;
    4) la especificidad, en porcentaje, de las predicciones (negativas) de la clase "non-allergen" correctamente clasificadas;
    5) el valor de PPV, en porcentaje
    6) el score F1, en porcentaje
    7) el valor de MCC, en porcentaje
    '''

    tp=0 #total de alergenos predecidos correctamente como "allergen" (True positives)
    tn=0 #total de no alergenos predecidos correctamente como "non-allergen" (True negatives)
    fn=0 #total de alergenos predecidos incorrectamente como "non-allergen" (False negatives)
    fp=0 #total de no alergenos predecidos incorrectamente como "allergen" (False positives)

    for i in range(len(predicted_class)):
        correct_clas=test_classif[i]
        if correct_clas=="allergen" and correct_clas==predicted_class[i]:
            tp+=1
        elif correct_clas=="non-allergen" and correct_clas==predicted_class[i]:
            tn+=1
        elif correct_clas=="allergen" and predicted_class[i]=="non-allergen":
            fn+=1
        elif correct_clas=="non-allergen" and predicted_class[i]=="allergen":
            fp+=1

    sensitivity=tp/(tp+fn) if tp+fn != 0 else 0
    specificity=tn/(tn+fp) if tn+fp != 0 else 0
    accuracy=(tp+tn)/(tp+fp+tn+fn) if tp+fp+tn+fn!=0 else 0
    ppv=tp/(tp+fp) if tp+fp!=0 else 0
    f1Score=(2*sensitivity*ppv)/(sensitivity+ppv) if sensitivity+ppv!=0 else 0
    mcc=((tp*tn)-(fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))!=0 else 0

    return accuracy*100, sensitivity*100, specificity*100, ppv*100, f1Score*100, mcc*100


def maxClasification(prediction, posClass="allergen", negClass="non-allergen"):
    '''
    Entrada:
    :param prediction: lista con las predicciones de las clasificaciones realizadas
    :param posClass: etiqueta de la clase positiva (por defecto: "allergen")
    :param negClass: etiqueta de la clase negativa (por defecto: "non-allergen")

    Salida:
    :return devuelve la clasificacion predominante, en caso de empate devuelve "allergen"
    '''
    r="allergen"
    c1=0
    c2=0
    for p in prediction:
        if p==posClass:
            c1+=1
        else:
            c2+=2
    if c2 > c1:
        r=negClass

    return r


###
### Test 1: conjunto de test positivos CDEP y matriz blosum 50
### >>> ks,ms,cas=draw3DGraphics("a_cdep.txt", "a_cden.txt", "a_cdep.txt", test_class="allergen")
### Mejor clasificación: 96.70%, k=3, m=1
###
### Test 2: conjunto de test positivos CDEP y matriz identidad
### >>> ks,ms,cas=draw3DGraphics("a_cdep_idn.txt", "a_cden_idn.txt", "a_cdep_idn.txt", test_class="allergen")
### Mejor clasificación: 96.70%, k=3, m=1
###
### Test 3: conjunto de test normal CDPN y matriz blosum 50
### >>> ks,ms,cas=draw3DGraphics("a_cdep.txt", "a_cden.txt", "a_cdpn.txt", test_class="non-allergen")
### Mejor clasificación: 100.00%, k=13, m=1
###
### Test 4: conjunto de test normal CDPN y matriz identidad
### >>> ks,ms,cas=draw3DGraphics("a_cdep_idn.txt", "a_cden_idn.txt", "a_cdpn_idn.txt", test_class="non-allergen")
### Mejor clasificación: 100.00%, k=5, m=1

def draw3DGraphics(pos_t_filename="a_cdep.txt", neg_t_filename="a_cden.txt", test_filename="a_cdp.txt"
                   , test_class="non-allergen", maxK=15, maxM=5, crossVal=False, test_length=0.1, holdOutTest=200
                   , leaveOneOut=False, kFolds=5, method="knn", params=[]):
    '''
    Entrada:
    :param pos_t_filename: archivo del conjunto de entrenamiento positivo.
    :param neg_t_filename: archivo del conjunto de entrenamiento negativo.
    :param test_filename: archivo del conjunto de prueba.
    :param test_class: (allergen o non-allergen) es la clase del conjunto de datos de test
    :param maxK: maximo valor de k (k vecinos mas cercanos)
    :param maxM: maximo valor de m (numero de alineaciones con mayor score)
    :param crossVal: True si queremos usar la validación cruzada
    :param test_length: (entre 0 y 1) longitud del conjunto de test para realizar la validación cruzada
    :param leaveOneOut: True si queremos usar la validación Leave-one-out
    :param method: la técnica de Aprendizaje Automático que se va a utilizar. Los valores aceptados son:
    "knn" (k Nearest Neighbors), "dt" (Decision Tree), "nb" (Naïve Bayes), "rf" (Random forest), "mlp" (Multilayer perceptron)
    :param params: lista de parámetros para el método dado

    Salida:
    Dibuja una grafica en 3D en la que se muestran los rendimientos alcanzados por el algoritmo kNN
    y diferentes configuraciones del parametro k (k vecinos) y m (numero de alineaciones con mayor score)
    que varian desde k=3 hasta kMax y desde m=1 hasta mMax.
    Se pinta en pantalla tambien la mejor configuracion de dichos parametros k y m
    que proporcionan el maximo rendimiento.
    '''
    xs=[] #puntos de la coordenada X
    ys=[] #puntos de la coordenada Y
    zs=[] #puntos de la coordenada Z
    features=[] #caracteristicas del conjunto de entrenamiento
    classifs=[] #clasificaciones del conjunto de entrenamiento
    testFeatures=[] #caracteristicas del conjunto de prueba
    testClassifs=[] #clasificaciones del conjunto de prueba

    #obtenemos las caracteristicas (score y longitud) por cada valor de m (mejores alineaciones, con mejor score)
    for m in range(1, maxM+1):
        '''if webApp:
            create_alignments_files(aligPos=False
                , aligNeg=False, aligTest=True, testSecFile=testSecFile)
        '''
        featToExtract=[True, True]
        f,c,ft,ct,p,pt = extract_all_features_and_classifications(featToExtract, m
                ,pos_t_filename, neg_t_filename, test_filename, test_class)
        
        #fa,a,_=get_features(pos_t_filename, m, "allergen")
        #fn,na,_=get_features(neg_t_filename, m, "non-allergen")
        #ft,nat,_=get_features(test_filename, m, test_class)
        #X=f,c
        #y=a+na
        features.append(f)
        testFeatures.append(ft)
        classifs.append(c)
        testClassifs.append(ct)

    #si el metodo es kNN
    if method=="knn":
        params=[[k,m] for k in range(3, maxK+2, 2) for m in range(1, maxM+1)]
        #calculamos el rendimiento de la clasificacion con kNN por cada par [k, m]
        for k,m in params:
            f,c,ft,ct,cp,ac,se,sp,ppv,f1Score,mcc=perform_prediction(method, features[m-1], classifs[m-1], testFeatures[m-1], testClassifs[m-1], k, crossVal, test_length, holdOutTest, leaveOneOut, kFolds, params={"k":k})
            xs.append(k)
            ys.append(m)
            zs.append(ac)
    else:
        for k,m in params:
            f,c,ft,ct,cp,ac,se,sp,ppv,f1Score,mcc=perform_prediction(method, features[m-1], classifs[m-1], testFeatures[m-1], testClassifs[m-1], k, crossVal, test_length, holdOutTest, leaveOneOut, kFolds)
            xs.append(se)
            ys.append(m)
            zs.append(ac)
        
            

        i=zs.index(max(zs)) #indice del mayor porcentaje de clasificación correcta
        print("Mejor clasificación: Accuracy = "+str("%.2f" % zs[i])+"%, k ="+str(xs[i])+", m ="+str(ys[i]))

    fig = plt.figure()
    axes = fig.gca(projection='3d')

    axes.set_xlim([3,maxK+1])
    axes.set_ylim([1,maxM])
    axes.set_zlim([min(zs), max(zs)])

    axes.invert_xaxis()
    axes.invert_yaxis()
    axes.set_xlabel("k")
    axes.set_ylabel("m")
    axes.set_zlabel("CA %")

    axes.plot_trisurf(xs, ys, zs, linewidth=0.2, color='#00feff')
    plt.show()


def drawSimpleFigure(xs, ys, figIindex=0, compLimits=False, nxs=None, nys=None, txs=None, tys=None, color=None
                     , plot=False, title=""):
    '''
    Entrada:
    :param xs: lista de coordenadas X
    :param ys: lista de coordenadas Y
    :param figIindex: indide de la figura a dibujar
    :param compLimits: True para hacer la comparativa con el estudio de Zorzet et al. 2002 con los mismos límites
    :param nxs: lista de coordenadas X de un segundo grupo a dibujar en la misma figura
    :param nys: lista de coordenadas Y de un segundo grupo a dibujar en la misma figura
    :param txs: lista de coordenadas X de un tercer grupo a dibujar en la misma figura
    :param tys: lista de coordenadas X de un tercer grupo a dibujar en la misma figura
    :param color: color para pintar los puntos dados por las coordenadas de xs e ys
    :param plot: True para dibujar la grafica, False no dibujarla aqui. Segun Matlabplot
    se procede preparando primero todas las figuras y finalmente se dibujan. Por ello,
    damos la opcion de dibujar una unica figura con esta funcion o dejar la responsabilidad
    al programa que utilice esta funcion de dibujar con matplotlib.pyplot.show() desde fuera.

    Salida:
    Prepara una figura para ser dibujada en pantalla segun las listas de puntos dadas.
    Si se indica plot=True, se muestra la figura.
    '''
    plt.figure(figIindex)
    plt.xlabel("Puntuación de la alineación")
    plt.ylabel("Longitud de la alineación")
    plt.title(title)

    if compLimits:
        plt.xlim(-200, 3400)
        plt.ylim(-50, 650)
    '''else:
        plt.xlim(-500, max(xs)+500)
        plt.ylim(-100, max(ys)+200)'''

    if nxs != None and nys != None:
        plt.scatter(nxs, nys, color="green")
        plt.scatter(xs, ys, color="red")

        if txs != None and tys != None:
            plt.scatter(txs, tys, color="blue")
    else:
        plt.scatter(xs, ys, color=color)

    if plot:
        plt.show()


def predict(X_train=[], y_train=[], protInfo_train=[], X_test=[], y_test=[], protInfo_test=[]
            , posAlFile="a_cdep.txt"
            , negAlFile="a_cden.txt"
            , testSecFile="created_test.fasta"
            , testAlFile="a_cdp.txt"
            , testClass="allergen"
            , featToExtract=[True,True]
            , method="knn"
            , params={'algorithm': 'auto', 'metric': 'minkowski', 'weights': 'distance', 'p': 3
                , 'leaf_size': 1, 'n_neighbors': 7}
            , plotAIO=False
            , webApp=False
            , plotPosAlgn=False, plotNegAlgn=False, plotTestAlgn=False
            , plotSurface=False, m=1, crossVal=False, testLength=0.05
            , holdOutTest=200, leaveOneOut=False, compLimits=False
            , showAllPredictions=False, kFolds=0, nExperiments=1, plotModel=False, figSameColor=False):
    '''
    Descripción:
    Programa principal, encargado de extraer las alineaciones de los conjuntos de datos usados,
    realizar la predicción de la clasificación del conjunto de pruebas, mostrar como resultado
    el rendimiento de una correcta clasificación de ese conjunto y, por último, si se quisiera
    dibujar las gráficas del resultado de esas alineaciones.

    Entrada:
    :param X_train: características del conjunto train (vacía por defecto, para extraerlas desde archivo)
    :param y_train: las clases del conjunto train
    :param protInfo_train: información sobre las proteínas del conjunto train
    :param X_test: características del conjunto de test (vacía por defecto, para extraerlas desde archivo)
    :param y_test: las clases del conjunto de test
    :param protInfo_test: información sobre las proteínas del conjunto test
    :param aPosFilename: archivo FASTA del conjunto de entrenamiento positivo del alineamiento.
    :param aNegFilename: archivo FASTA del conjunto de entrenamiento negativo del alineamiento.
    :param aTestFilename: archivo FASTA del conjunto de prueba del alineamiento.
    :param testClass: (allergen o non-allergen) es la clase del conjunto de datos de test
    :param featToExtract: características para extraer de los archivos de alineamiento.
    Se trata de una lista en la que en cada posición se indica si una característica
    se extrae o no, indicando True en caso afirmativo, y False en caso contrario.
    A continuación se indican qué características se interpretan en cada posición de esta lista,
    que, además, en caso que la lista se indique vacía, automáticamente se extraerán siempre
    las dos primeras características, es decir, Smith-Waterman Score y Alignment Length.

    [0: Smith-Waterman Score, 1: Alignment Length, 2: % Identity, 3: % Similirity, 4: Z-Score,
    5: initn, 6: init1, 7: bits, 8: E value]

    :param method: la técnica de Aprendizaje Automático que se va a utilizar. Los valores aceptados son:
    "knn" (k Nearest Neighbors), "dt" (Decision Tree), "nb" (Naïve Bayes), "rf" (Random forest)
    , "mlp" (Multilayer perceptron), "rbm" (Bernoulli Restricted Boltzmann Machine)
    :param posSecsFilename: (Defecto "": no alinea) archivo FASTA del conjunto de secuencias del entrenamiento positivo.
    :param negSecsFilename: (Defecto "": no alinea, Requiere que posSecsFilename != "") archivo FASTA del conjunto de secuencias del entrenamiento negativo.
    :param testSecsFilename: (Defecto "": no alinea, Requiere que posSecsFilename != "") archivo FASTA del conjunto de secuencias de prueba.
    :param aligPos: True para alinear las secuencias positivas indicadas en el archivo posSecsFilename consigo mismo
    :param aligNeg: True para alinear las secuencias negativas indicadas en el archivo negSecsFilename con las del archivo posSecsFilename
    :param plotAIO: True para dibujar la gráfica de todas las alineaciones en una sola
    :param webApp: True si es la aplicacion web la que llama a este metodo
    :param plotPosAlgn: True para dibujar la gráfica de las alineaciones del conjunto de entrenamiento positivo
    :param plotNegAlgn: True para dibujar la gráfica de las alineaciones del conjunto de entrenamiento negativo
    :param plotTestAlgn: True para dibujar la gráfica de las alineaciones del conjunto de prueba
    :param plotSurface: True para dibujar la gráfica donde se muestran los diferentes rendimientos conseguidos variando k y m
    :param m: número total de alineaciones a considerar de entre las que tienen mejor puntuación de alineación
    :param crossVal: True si queremos usar la validación cruzada
    :param testLength: (entre 0 y 1) longitud del conjunto de test para realizar la validación cruzada
    :param leaveOneOut: True si queremos usar la validación Leave-one-out
    :param compLimits: True para hacer la comparativa con el estudio de Zorzet et al. 2002 con los mismos límites
    :param kFolds: k folds (divisiones) para realizar la validación cruzada
    :param plotModel: si es True imprime una figura del modelo
    :param figSameColor: si es True imprime la figura con los alérgenos del mismo color que los no alérgenos y el
    conjunto de test.

    Salida:
    :return Devuelve la tupla ([clasificacion predicha], [nombres de las proteínas])
    '''

    if webApp:
        print("Creating test alignment file")
        create_alignments_files(aligPos=False
            , aligNeg=False, aligTest=True, testSecFile=testSecFile)

    if X_train==[] and y_train==[] and protInfo_train==[] and X_test==[] and y_test==[] and protInfo_test==[]:
        f,c,ft,ct,p,pt = extract_all_features_and_classifications(featToExtract, m,posAlFile, negAlFile
                                                              , testAlFile, testClass)
    else:
        f, c, ft, ct, p, pt = X_train, y_train, X_test, y_test, protInfo_train, protInfo_test

    #realizar la predicción
    print("Predicting...")
    f,c,ft,ct,cp,ac,se,sp,ppv,f1Score,mcc=perform_prediction(method, f, c, ft, ct, kFolds, params, plotModel=plotModel)

    #Presentar los resultados
    # print("Resultados")

    if showAllPredictions:
        print("Predicción de la clasificación: "+str(cp))

    #dibujar las gráficas 3D que muestran los rendimientos diferentes variando k y m
    if plotSurface:
        draw3DGraphics(posAlFile, negAlFile, testAlFile, testClass, crossVal=crossVal, test_length=testLength
                       , holdOutTest=holdOutTest, leaveOneOut=leaveOneOut, kFolds=kFolds, method=method
                       , nExperiments=nExperiments)

    #dibujar las gráficas 2D de las alineaciones de los conjuntos de datos
    if plotAIO or plotPosAlgn or plotNegAlgn or plotTestAlgn:
        if plotAIO or plotPosAlgn:
            xs=[item[0] for i,item in enumerate(f) if c[i]=="allergen"]
            ys=[item[1] for i,item in enumerate(f) if c[i]=="allergen"]
        if plotAIO or plotNegAlgn:
            nxs=[item[0] for i,item in enumerate(f) if c[i]=="non-allergen"]
            nys=[item[1] for i,item in enumerate(f) if c[i]=="non-allergen"]
        if plotAIO or plotTestAlgn:
            txs=[item[0] for item in ft]
            tys=[item[1] for item in ft]
    i=0
    if plotAIO:
        drawSimpleFigure(xs, ys, i, compLimits, nxs, nys, txs, tys)

    if plotPosAlgn:
        i+=1
        col='red' if not figSameColor else 'blue'
        drawSimpleFigure(xs, ys, i, compLimits, color=col, title="Alérgenos")

    if plotNegAlgn:
        i+=1
        col = 'green' if not figSameColor else 'blue'
        drawSimpleFigure(nxs, nys, i, compLimits, color=col, title="No alérgenos")

    if plotTestAlgn:
        i+=1
        col = 'orange' if not figSameColor else 'blue'
        drawSimpleFigure(txs, tys, i, compLimits, color=col, title="Test")

    plt.show()

    #ordenamos primero apareceran todos los alérgenos y luego todos los no alérgenos
    cp_indexes = list(range(len(cp))) #para ello usaremos los índices de la list de la clasificación predicha
    cp_indexes.sort(key=cp.__getitem__) #la ordenación devuelve la lista de los índices ordenados
    sorted_cp = list(map(cp.__getitem__, cp_indexes))
    sorted_pt = list(map(pt.__getitem__, cp_indexes))

    return sorted_cp, sorted_pt

def show_learning_methods_performance(featToExtract=[True, True], method="knn"
    , params={"k":3}, kFolds=5, m=1
    , posAlFile="a_cdep.txt", negAlFile="a_cden.txt", printNativeClassReport = False):
    '''
    Entrada:
    :param printNativeClassReport: True para imprimir el informe de la clasificacion que
    proporciona scikitlearn.
    
    Salida:
    Muestra por pantalla la precision alcanzada en la clasificacion correcta
    para el metodo indicado.
    '''

    print("Valorando Método "+str(method))
    print("Extrayendo features..")
    f,c,_,_,_,_ = extract_all_features_and_classifications(featToExtract, m, posAlFile, negAlFile)

    #realizar la predicción
    print("Training performance...")
    _,_,_,_,_,ac,se,sp,ppv,f1Score,mcc=perform_prediction(method, f, c, [], [], kFolds, params
                                                          , printNativeClassReport)
    
    #Presentar los resultados
    print("\nValoración del Método "+str(method))
    print("Con m="+str(m)+" mejores alineamientos, kFolds="+str(kFolds))
    print("Características extraídas: "+features_for_extracting_to_string(featToExtract))

    if method=="knn":
        k=params.get("k")
        print("Y con k="+str(k)+"\n")

    print("Accuracy = "+str("%.2f" % ac)+"%, Sensitivity = "+str("%.2f" % se)+"%"
        +", Specification = "+str("%.2f" % sp)+"%"+", ppv = "+str("%.2f" % ppv)+"%"
        +", F1 = "+str("%.2f" % f1Score)+"%"+", MCC = "+str("%.2f" % mcc)+"%")
    print(str("_"*10)+"\n")

    return ac, se, sp, ppv, f1Score, mcc


def show_all_learning_methods_performance(methods=[("mlp",{}), ("log", {}), ("rf", {}), ("knn", {"k":3})
    , ("dt", {}), ("nb", {}), ("svm", {}), ("rbm", {})]
    , featToExtract = [True,True,True,True,True,True,True,True,True,True]
    , kFolds=5, m=1):
    '''
    Entrada:
    :param featToExtract: características para extraer de los archivos de alineamiento.
    Se trata de una lista en la que en cada posición se indica si una característica
    se extrae o no, indicando True en caso afirmativo, y False en caso contrario.
    A continuación se indican qué características se interpretan en cada posición de esta lista,
    que, además, en caso que la lista se indique vacía, automáticamente se extraerán siempre
    las dos primeras características, es decir, Smith-Waterman Score y Alignment Length.

    [0: Smith-Waterman Score, 1: Alignment Length, 2: % Identity, 3: % Similirity, 4: Z-Score,
    5: initn, 6: init1, 7: bits, 8: E value]

    :param methods: lista de tuplas [(nombre del metodo, {diccionario con los parametros del metodo})]
    
    Salida:
    Metodo todo en uno para mostrar la precision de todos los metodos que queramos
    de una sola vez.
    '''
    for met in methods:
        show_learning_methods_performance(featToExtract, met[0], met[1], kFolds, m)


def plot_decision_tree(model, filename='dt_plot'):
    '''
    Muestra el grafo de un árbol de decisión
    :param model: el modelo del árbol de decisión
    :param filename: el path del archivo donde se guardará
    '''
    workingDir = os.path.dirname(__file__)
    alignsPath = '/alignments/'
    if os.path.exists(workingDir + alignsPath):
        path = workingDir + alignsPath + filename
    else:
        path = alignsPath + filename

    str_buffer = StringIO()
    tree.export_graphviz(model, out_file=str_buffer)
    graph = pydotplus.graph_from_dot_data(str_buffer.getvalue())
    graph.write_png(path+".png")

def prediction_test(method, params, m=1, feats=[True, True]):
    print("Test con " + method + ", m=" + str(m) + " y feats=" + str(features_for_extracting_to_string(feats)))

    # test de alérgenos
    cp, pt = predict(method=method, params=params, m=m, featToExtract=feats, webApp=False, testAlFile="a_cdp.txt")
    counter = Counter(cp)
    # print(str(counter))
    set = (counter['allergen'] * 100) / len(cp)

    # test de no alérgenos
    cp, pt = predict(method=method, params=params, m=m, featToExtract=feats, webApp=False, testAlFile="a_cdpn.txt")
    counter = Counter(cp)
    # print(str(counter))
    spt = (counter['non-allergen'] * 100) / len(cp)
    act = (set + spt) / 2

    print("With our test data set >> Accuracy = " + str("%.2f" % act) + "%, Sensitivity = " + str("%.2f" % set)
          + "%" + ", Specification = " + str("%.2f" % spt) + "%")

    print("With CV (k-fold=10) + Stratification")
    ac, se, sp, ppv, f1Score, mcc = show_learning_methods_performance(feats, method=method, params=params,
                                                                      kFolds=10)

    return act, set, spt, ac, se, sp, ppv, f1Score, mcc

def tuning_model_performance(method, combs=[],score='precision', minM=1, maxM=5, reduction=0, verbose=False):
    featCombs = [[True], [False, True], [True, True], [False, False, True], [False, False, False, True]
        , [True, False, True], [True, True, True], [True, True, False, True], [True, True, True, True]
        , [False, False, True, True], [True, False, False, True]] if len(combs) == 0 else combs


    bests = []

    print("Tuning model: " + method)
    for m in range(max(min(minM, 9), 1), max(min(maxM+1, 10), 1)):
        for comb in featCombs:
            if verbose:
                print("checking with m=" + str(m) + " and feats: " + str(comb))
            bestParams, _, _, _, _, _, _ = tuning_model(method, score, comb, m, reduction=reduction)

            if verbose:
                print("best params: " + str(bestParams) + "\n")
            act, set, spt, ac, se, sp, ppv, f1Score, mcc = prediction_test(method, bestParams, m, comb)
            bests.append({"m": m, "feats": comb, "params": bestParams, "act": act, "set": set, "spt": spt
                         , "ac": ac, "se": se, "sp": sp, "ppv": ppv, "f1s": f1Score, "mcc": mcc})
        #por cada m vamos presentando resultados parciales
        print_best_tuning_params(bests)

    #presentamos el resultado final
    print_best_tuning_params(bests)

def print_best_tuning_params(bests):
    maxTestAccuracy = sys.float_info.min
    maxCVAccuracy = maxTestAccuracy
    maxTestParams = None
    maxParams = None
    for b in bests:
        if b["act"] > maxTestAccuracy:
            maxTestAccuracy = b["act"]
            maxTestParams = b
        if b["ac"] > maxCVAccuracy:
            maxCVAccuracy = b["ac"]
            maxParams = b

    print("Mejor tuning para CV: " + str(maxParams))
    print("Accuracy = " + str("%.2f" % maxParams["ac"]) + "%, Sensitivity = " + str("%.2f" % maxParams["se"])
          + "%" + ", Specification = " + str("%.2f" % maxParams["sp"]) + "%")
    print("Mejor tuning con conjunto de prueba propio: " + str(maxTestParams))
    print("Accuracy = " + str("%.2f" % maxTestParams["act"]) + "%, Sensitivity = " + str(
        "%.2f" % maxTestParams["set"])
          + "%" + ", Specification = " + str("%.2f" % maxTestParams["spt"]) + "%")
    print("Todos los tunings:")
    print(bests)