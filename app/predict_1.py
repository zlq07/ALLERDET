#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------
UNIVERSIDAD DE SEVILLA
ESCUELA TÉCNICA SUPERIOR DE INGENIERÍA INFORMÁTICA
GRADO INGENIERÍA INFORMÁTICA – TECNOLOGÍAS INFORMÁTICAS

Trabajo fin de grado: Predicción de la alergenicidad potencial de las proteínas alimentarias
mediante diferentes técnicas de Aprendizaje Automático.
Autor: Francisco Manuel García Moreno
Tutor: Miguel Ángel Gutiérrez Naranjo
Fecha: 30-06-2016 (ultima modificacion: 27/04/2017)
------------------------------------------------------------------------------------------------
"""

import os
import os.path
import math
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import alignments


def create_alignments_files(aligPos=True, aligNeg=True, aligTest=False
                            , posSecFile="reduced_all_allergens.fasta"
                            , posAlFile="a_cdep.txt"
                            , negSecFile="reduced_all_nonallergens.fasta"
                            , negAlFile="a_cden.txt"
                            , testSecFile="created_test.fasta"
                            , testAlFile="a_cdp.txt"
                            , alignsPath="/alignments/"
                            , fastaProgramPath="/alignments/fasta-36.3.8d/bin/fasta36"
                            , fParam="-12", gParam="-2", verbose=True):
    '''
    Crea los archivos de alineamiento con el programa FASTA36

    Prerequisitos:
    - Disponer del programa FASTA36 en el sistema operativo

    Entrada:
    :param  aligPos: True para alinear las secuencias positivas.
    :param  aligNeg: True para alinear las secuencias negativas.
    :param  aligTest: True para alinear las secuencias de prueba.
    :param  posSecFile: nombre del archivo TXT del alineamiento del conjunto de
    entrenamiento positivo obtenido con el programa FASTA36.
    :param  posAlFile: nombre del archivo FASTA del conjunto de secuencias positivo.
    :param  posSecFile: nombre del archivo FASTA del conjunto de secuencias negativo.
    :param  negAlFile: nombre del aarchivo TXT del alineamiento del conjunto de
    entrenamiento negativo obtenido con el programa FASTA36.
    :param  testSecFile: nombre del archivo FASTA con secuencias del conjunto test.
    :param  testAlFile: nombre del archivo TXT del alineamiento del conjunto de
    prueba obtenido con el programa FASTA36.
    :param  fParam: parámetro f del programa FASTA36
    :param  gParam: parámetro g del programa FASTA36
    :param  verbose: True para obtener por consola logs del proceso.
    '''
    workingDir = os.path.dirname(__file__)

    if os.path.exists(workingDir+alignsPath):
        alignsPath=workingDir+alignsPath
        fastaProgramPath=workingDir+fastaProgramPath
    else:
        print("no existe path: "+workingDir+alignsPath)

    #es necesario el archivo de alineamiento positivo
    if alignsPath+posSecFile != "":
        #crear archivo del alineamiento positivo
        if aligPos:
            cmd=fastaProgramPath+" -f "+fParam+" -g "+gParam+" "+alignsPath+posSecFile+" "+alignsPath+posSecFile\
                +" > "+alignsPath+posAlFile
            if verbose:
                print("Ejecutando comando: "+cmd)
            os.system(cmd)
            if verbose:
                print("Se ha creado correctamente el archivo del alineamiento"
                " del conjunto de datos entrenamiento positivo")

        #crear archivo del alineamiento negativo
        if alignsPath+negSecFile != "" and aligNeg:
            cmd=fastaProgramPath+" -f "+fParam+" -g "+gParam+" "+alignsPath+negSecFile+" "+alignsPath+posSecFile\
                +" > "+alignsPath+negAlFile
            if verbose:
                print("Ejecutando comando: "+cmd)
            os.system(cmd)
            if verbose:
                print("Se ha creado correctamente el archivo del alineamiento"
                " del conjunto de datos entrenamiento negativos")

        if aligTest and alignsPath+testSecFile != "":
            if os.path.exists(alignsPath+testSecFile):
                cmd=fastaProgramPath+" -f "+fParam+" -g "+gParam+" "+alignsPath+testSecFile+" "+alignsPath+posSecFile\
                    +" > "+alignsPath+testAlFile
                if verbose:
                    print("Ejecutando comando: "+cmd)
                os.system(cmd)
            else:
                #si la consulta se realiza mediante un texto con formato FASTA
                #y no mediante un fichero .fasta se crea dicho fichero (w+)
                #y después se alinea este .fasta
                test_created_file = alignsPath+"created_test.fasta"
                open(test_created_file, "w+").write(testSecFile)
                cmd=fastaProgramPath+" -f "+fParam+" -g "+gParam+" "+test_created_file+" "+alignsPath+posSecFile\
                    +" > "+alignsPath+testAlFile
                if verbose:
                    print("Ejecutando comando: "+cmd)
                os.system(cmd)
                if verbose:
                    print("Se ha creado correctamente el archivo .fasta del"
                    " conjunto de datos de prueba")
            if verbose:
                print("Se ha creado correctamente el archivo del alineamiento"
                " del conjunto de datos de prueba")

        posAlFile=alignsPath+posAlFile
        negAlFile=alignsPath+negAlFile
        testAlFile=alignsPath+testAlFile

    return posAlFile, negAlFile, testAlFile

def splitFastaSeqs(filename):
    '''
    Extrae de un archivo FASTA sus secuencias
    :param filename: el path del archivo FASTA
    :return: una tupla compuesta por una lista de los identificadores de las secuencias y una lista de las secencias
    '''
    seqIds = []
    seqs = []

    with open(filename, "rt") as f:
        content = f.read()
        lines = content.split('\n')
        seq = ''
        for l in lines:
            l = l.strip()
            m=l.split(">")
            if(len(m)>1):
                seqIds.append(m[1])
                if seq != '':
                    seqs.append(seq)
                    seq = ''
            elif(len(m) == 1):
                seq += l
    return seqIds, seqs

def how_many_seqs_from_a_are_duplicated_in_b(filenameA, filenameB):
    '''
    Encuentra las secuencias del archivo A que están duplicadas en el archivo B si las hubiese.

    :param filenameA: el archivo donde se quiere ver si existen secuencias duplicadas en el archivo B
    :param filenameB: el archivo base donde tenemos nuestras secuencias validas
    :return: lista coon las secuencas del archivo A que se han encontrado duplicadas en el archivo B
    '''
    sidA, seqsA = splitFastaSeqs(filenameA)
    sidB, seqsB = splitFastaSeqs(filenameB)
    duplicated = []

    for i, a in enumerate(seqsA):
        if a in seqsB:
            duplicated.append([sidA[i], a])
    return duplicated

def create_fasta_file_without_duplications(filenames, resultFilename='alignments/result.fasta', seqsNotIn=[]):
    '''
    Crea un fichero con extension .fasta sin incluir duplicaciones de secuencias a partir de las secuencias
    extraidas de los archivos FASTA proporcionados.
    :param filenames: una lista de paths de los archivos FASTA de los que se desea extraer sus secuencias
    :param resultFilename: el path donde se guardará el archivo final resultante
    :param seqsNotIn: (opcional) lista de secuencias que no se quieran duplicar
    '''
    res=[]
    for f in filenames:
        sid, seqs = splitFastaSeqs(f)
        c = Counter(seqs)

        for s in c.keys():
            if not any(s==x[1] for x in res) and s not in seqsNotIn:
                i=seqs.index(s)
                res.append([sid[i], s])
    content = ""
    for i,r in enumerate(res):
        if i>0:
            content += "\n"
        content += ">" + r[0] + "\n" + r[1]
    with open(resultFilename, "w") as f:
        f.write(content)



def extract_all_features_and_classifications(featToExtract=[True, True], m=1
    , posAlFile="", negAlFile="", testAlFile="", testClass="allergen"
    , appFolder="/app/", alignsPath="/alignments/"):

    '''
    Entrada:
    :param featToExtract: características para extraer. Se trata de una lista en la
    que en cada posición se indica si una característica se extrae o no,
    indicando True en caso afirmativo, y False en caso contrario. A continuación
    se indican qué características se interpretan en cada posición de esta
    lista, que en caso que la lista se indique vacía, automáticamente se
    extraerán siempre las dos primerar características, es decir, Smith-Waterman
     Score y Alignment Length.

    [0: Smith-Waterman Score, 1: Alignment Length, 2: % Identity
    , 3: % Similirity, 4: Z-Score, 5: initn, 6: init1, 7: bits, 8: E value]

    :param m: número de alineaciones con la mejor puntuación de alineación
    ( f,c,ft,ct,pi,pit): lista con las features del conjunto entrenamiento,
    lista de clasificaciones del conjunto entrenamiento, lista de features
    del conjunto de test, lista de clasificaciones del conjunto de test,
    lista de informacion de las proteinas del conjunto de entrenamiento,
    lista de informacion de las proteinas del conjunto de test.

    Salida:
    :return Tupla: (train features, train classification, test features
    , test classification, informacion de las proteinas del train
    , informacion de las proteinas del test)
    '''
    workingDir = os.path.dirname(__file__)

    if os.path.exists(workingDir+alignsPath):
        path=workingDir+alignsPath
    else:            
        path = alignsPath
        
    f1,f2,c1,c2,ft,ct,p1,p2,pi,pit=[],[],[],[],[],[],[],[],[],[]

    #características y clasificaciones del conjunto de entrenamiento "allergen"
    if posAlFile != "":
        f1,c1,p1=get_features(path+posAlFile, m, "allergen", featToExtract)

    #características y clasificaciones del conjunto de entrenamiento "non-allergen"
    if negAlFile != "":
        f2,c2,p2=get_features(path+negAlFile, m, "non-allergen", featToExtract)

    #caracteristicas y clasificaciones del conjunto de prueba
    if testAlFile != "":
        ft,ct,pit=get_features(path+testAlFile, m, testClass, featToExtract)

    f=f1+f2 #caracteristicas del conjunto de entrenamiento
    c=c1+c2 #clasificaciones del conjunto de entrenamiento
    pi=p1+p2

    # eliminar elementos repetidos del conjunto de entrenamiento
    d=[]
    for i,x in enumerate(f):
        if x in d:
            f.pop(i)
            c.pop(i)
            pi.pop(i)
        else:
            d.append(x)
    print("train " + str(len(f)) + "test "+str(len(ft)))
    return f,c,ft,ct,pi,pit



def get_features(filename, m, classType, featToExtract=[True, True]):
    '''
    Entrada:
    :param filename: nombre del fichero que contiene las alineaciones a procesar
    :param m: número de alineaciones con la mejor puntuación de alineación
    :param classType: la clase que se asigna al conjunto de datos, en nuestro caso
    son dos valores posibles: allergen o non-allergen.
    :param featToExtract: características para extraer. Se trata de una lista en la que
    en cada posición se indica si una característica se extrae o no, indicando True
    en caso afirmativo, y False en caso contrario. A continuación se indican qué
    características se interpretan en cada posición de esta lista, que en caso que la lista
    se indique vacía, automáticamente se extraerán siempre las dos primerar características,
    es decir, Smith-Waterman Score y Alignment Length.

    [0: Smith-Waterman Score, 1: Alignment Length, 2: % Identity, 3: % Similirity, 4: Z-Score,
    5: initn, 6: init1, 7: bits, 8: E value]

    Salida:
    Devuelve una tupla con los patrones, es decir, las características extraídas
    de las m alineaciones con mejor puntuación de alineación e información de las
    proteínas desde un archivo de alineaciones obtenido con el programa FASTA36.
    :return Tupla: (features, classfication, info de las proteinas)
    '''
    workingDir = os.path.dirname(__file__)

    if os.path.exists(workingDir+filename):
        filename = workingDir+filename

    content = ""
    with open(filename, "rt") as f:
        content = f.read()

        #f = open(filename, 'rt')
        #content = f.read()
        presec = content.split(">>>")[1:] #excluimos la parte inicial del fichero
        aligmentsNumbers=[]
        features=[]
        classif=[]
        protInfos=[]

        for alignment in presec:
            pro=alignment.split("\n")[0].split("|") #descripción de la proteína
            protInfo=[]
            protName=pro[0]

            #nombres de las proteínas
            if pro[0]=="gi":
                '''if pro[1]=="Q90YK8":
                    print(pro)'''
                protName=pro[1] if len(pro) > 1  else ""
                protName=pro[3] if len(pro) > 3  else protName
                protName=pro[4] if len(pro) > 4 else protName

            elif pro[0]=="sp" or pro[0]=="tr":
                p=pro[2].split(" ")
                protName=" ".join(p[1:]).split("PE")[0]

            if len(pro) > 1:
                protInfo=[pro[0],pro[1],protName]
            else:
                protInfo=[protName]
            protInfos.append(protInfo)

            #extracción de características
            ali = alignment.split(">>")#("Smith-Waterman score:")
            if len(ali) > 1:
                ali=ali[1:] #quitamos el resumen de los mejores alineamientos
                aBest=[]

                for a in ali:
                    firstLine=a.split("\n")[0]
                    #solo las 2 lineas del alineamiento donde estan las características
                    feats=" ".join(a.split("\n")[1:3])#(";") #excluyendo la primera (nombre de la proteina)
                    s=feats.split("Smith-Waterman score:")
                    score=int(s[1].split(";")[0].strip())

                    lengthStr = firstLine.split("aa)")[0].split("(")
                    length=lengthStr[1].strip()

                    if not length.isdigit():
                        length=lengthStr[len(lengthStr)-1].split("(")[0].strip();
                        if not length.isdigit():
                            length = length.split("a`)")[0]

                    length=int(length)#int(s[1].split(") in ")[1].split("aa")[0].strip())

                    identity=0
                    extIdentity=False #extraer identity
                    totalExtractions=len(featToExtract)

                    if totalExtractions > 2:
                        extractions=[]
                        extIdentity = totalExtractions >=3 and featToExtract[2] #extraer identity
                        if extIdentity:
                            identity=float(s[1].split("% identity")[0].split(";")[1].strip())
                        if not extIdentity or (extIdentity and identity >= 35): #FAO/WHO guidelines
                            extractions=[]
                            for i in range(len(featToExtract)):
                                extract=featToExtract[i]

                                if extract:
                                    if i==0: # Smith-Waterman Score
                                        feat=score
                                    elif i==1: # Alignment Length
                                        feat=length
                                    elif i==2: # % Identity
                                        feat = identity
                                    elif i==3: # % Similarity
                                        feat = float(s[1].split("% similar")[0].split("% identity (")[1].strip())
                                    elif i==4: # Z-Score
                                        feat = float(s[0].split("Z-score:")[1].split("bits")[0].strip())
                                    elif i==5: # initn
                                        feat = int(s[0].split("initn:")[1].split("init1")[0].strip())
                                    elif i==6: # init1
                                        feat = int(s[0].split("init1:")[1].split("opt")[0].strip())
                                    elif i==7: # bits
                                        feat = float(s[0].split("bits:")[1].split("E(")[0].strip())
                                    elif i==8: # E value
                                        feat = float(s[0].split("E(")[1].split("):")[1].strip())

                                    if feat != None:
                                        extractions.append(feat)
                        if extractions != []:
                            aBest.append(extractions)
                    else:
                        aBest.append([score,length])

                aBest.sort(reverse=True) #ordenamos de mayor a menor score
                features.extend(aBest[:m]) #los mejores m alineamientos segun el primer elemento

        classif=[classType]*len(features)

    return features, classif, protInfos


def features_for_extracting_to_string(featToExtract=[True, True]):
    '''
    Entrada:
    :param featToExtract: características para extraer. Se trata de una lista en la que
    en cada posición se indica si una característica se extrae o no, indicando True
    en caso afirmativo, y False en caso contrario. A continuación se indican qué
    características se interpretan en cada posición de esta lista, que en caso que la lista
    se indique vacía, automáticamente se extraerán siempre las dos primerar características,
    es decir, Smith-Waterman Score y Alignment Length.

    [0: Smith-Waterman Score, 1: Alignment Length, 2: % Identity, 3: % Similirity, 4: Z-Score,
    5: initn, 6: init1, 7: bits, 8: E value]

    Salida:
    :return Devuelve un string que indica las características para extraer separadas por
    comas. Ejemplo: "Smith-Waterman Score","Alignment Length" 
    '''
    res=[]
    for i in range(max(min(9, len(featToExtract)), 1)):#max de 9 features
        extract=featToExtract[i]

        if extract:
            if i==0: # Smith-Waterman Score
                feat="Smith-Waterman Score"
            elif i==1: # Alignment Length
                feat="Alignment Length"
            elif i==2: # % Identity
                feat = "Identity"
            elif i==3: # % Similarity
                feat = "Similarity"
            elif i==4: # Z-Score
                feat = "Z-Score"
            elif i==5: # initn
                feat = "initn"
            elif i==6: # init1
                feat = "init1"
            elif i==7: # bits
                feat = "bits"
            elif i==8: # E value
                feat = "E value"
            res.append(feat)
    return ",".join(res)


def create_prediction_model(method, params={}):
    '''
    Crea el modelo de predicción
    :param method: la técnica de aprendizaje que se desea utilizar
    :param features: lista de las características
    :param classifications: lista de las clases
    :param test_data: lista del conjunto de test
    :param tests_classif: lista de las clases del conjunto de test
    :param params: parámetros adicionales de configuración del modelo
    :return: Devuelve el modelo de predicción configurado, con los parámetros indicados
    '''
    model = None
    # eleccion del metodo de clasificación
    if method == "knn":
        distance = 'euclidean'
        k = 3
        param = params.get("k")
        if param != None and param > 2:
            k = param
        param = params.get("d")
        if param != None and param != "":
            distance = param
        model = KNeighborsClassifier(n_neighbors=k, metric=distance
                                     , weights='distance')  # el modelo kNN
    elif method == "dt":
        model = tree.DecisionTreeClassifier()
    elif method == "nb":
        model = GaussianNB()
    elif method == "rf":
        # http://stackoverflow.com/questions/20463281/how-do-i-solve-overfitting-in-random-forest-of-python-sklearn
        model = RandomForestClassifier(n_estimators=20, max_features=0.4)
    elif method == "mlp":
        model = MLPClassifier()
    elif method == "svm":
        model = svm.SVC()
    elif method == "log":
        model = linear_model.LogisticRegression(C=100.0)
    elif method == "rbm":
        logistic = linear_model.LogisticRegression(C=100)
        rbm = BernoulliRBM(learning_rate=0.1, n_components=100, n_iter=20, random_state=0)

        # finalmente, el clasificador
        model = Pipeline([("rbm", rbm), ("logistic", logistic)])
    return model

def perform_prediction(method, features, classifications, test_data
    , tests_classif, kFolds=0, params={}, printNativeClassReport=False):
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
            model.fit(featsTrain, classTrain)
            class_predicted=model.predict(featsTest)

            accuracy,sensitivity,specifity,ppv,f1Score,mcc=classification_performance(classTest, class_predicted)#model.score(X_test, y_test)
            measures.append([accuracy,sensitivity,specifity,ppv,f1Score,mcc])
            '''print("Accuracy = "+str("%.2f" % accuracy)+"%, Sensitivity = "+str("%.2f" % sensitivity)+"%"
                  +", Specification = "+str("%.2f" % specifity)+"%"+", ppv = "+str("%.2f" % ppv)+"%"
                  +", F1 = "+str("%.2f" % f1Score)+"%"+", MCC = "+str("%.2f" % mcc)+"%")'''

            if printNativeClassReport:
                print(classification_report(classTest, class_predicted, target_names=["allergen", "non-allergen"]))

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
            featsTrain.extend(feat_train)
            classTrain.extend(class_train)
            featsTest.append(feat_test[0])
            classTest.append(class_test[0])
            model = create_prediction_model(method, params)
            model.fit(feat_train, class_train)
            cpred=model.predict(feat_test)
            class_predicted.append(cpred)
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
    if test_data:
        featsTrain=features
        classTrain=classifications
        featsTest=test_data
        model = create_prediction_model(method, params)
        model.fit(featsTrain, classTrain)
        class_predicted=model.predict(featsTest)
        
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


def predict(posSecFile="reduced_all_allergens.fasta"
            , posAlFile="a_cdep.txt"
            , negSecFile="reduced_all_nonallergens.fasta"
            , negAlFile="a_cden.txt"
            , testSecFile="created_test.fasta"
            , testAlFile="a_cdp.txt"
            , testClass="allergen"
            , featToExtract=[True,True,True,True,True,True,True,True,True,True]
            , method="rf"
            , params={}
            , plotAIO=False
            , webApp=False
            , plotPosAlgn=False, plotNegAlgn=False, plotTestAlgn=False
            , plotSurface=False, m=1, crossVal=False, testLength=0.05
            , holdOutTest=200, leaveOneOut=False, compLimits=False
            , showAllPredictions=False, kFolds=5, nExperiments=1):
    '''
    Descripción:
    Programa principal, encargado de extraer las alineaciones de los conjuntos de datos usados,
    realizar la predicción de la clasificación del conjunto de pruebas, mostrar como resultado
    el rendimiento de una correcta clasificación de ese conjunto y, por último, si se quisiera
    dibujar las gráficas del resultado de esas alineaciones.

    Entrada:
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

    Salida:
    :return Devuelve la tupla ([clasificacion predicha], [nombres de las proteínas])
    '''

    if webApp:
        create_alignments_files(aligPos=False
            , aligNeg=False, aligTest=True, testSecFile=testSecFile)

    f,c,ft,ct,p,pt = extract_all_features_and_classifications(featToExtract, m,posAlFile, negAlFile
                                                              , testAlFile, testClass)

    #realizar la predicción
    f,c,ft,ct,cp,ac,se,sp,ppv,f1Score,mcc=perform_prediction(method, f, c, ft, ct, kFolds, params)

    #Presentar los resultados
    print("Resultados")

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
        drawSimpleFigure(xs, ys, i, compLimits, color='red', title="Alérgenos")

    if plotNegAlgn:
        i+=1
        drawSimpleFigure(nxs, nys, i, compLimits, color='green', title="No alérgenos")

    if plotTestAlgn:
        i+=1
        drawSimpleFigure(txs, tys, i, compLimits, color='blue', title="Test")

    plt.show()

    return cp,pt

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
        k=3
        param=params.get("k")
        if param > 2:
            k=3
        print("Y con k="+str(k)+"\n")

    print("Accuracy = "+str("%.2f" % ac)+"%, Sensitivity = "+str("%.2f" % se)+"%"
        +", Specification = "+str("%.2f" % sp)+"%"+", ppv = "+str("%.2f" % ppv)+"%"
        +", F1 = "+str("%.2f" % f1Score)+"%"+", MCC = "+str("%.2f" % mcc)+"%")
    print(str("_"*10)+"\n")


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
    


##----------------------------------------------------------------------------------------
## TESTS:
##----------------------------------------------------------------------------------------
allDataset = 'alignments/allerdictor/B/alg.fa'
allTestDataset = 'alignments/created_test.fasta'

show_learning_methods_performance([True,True,True,True,True,True,True,True,True,True]
    , method="mlp")

'''
show_learning_methods_performance([True,True,True,True,True,True,True,True,True,True]
    , method="mlp")
show_learning_methods_performance([True,True,True,True,True,True,True,True,True,True]
    , method="rf")
show_learning_methods_performance([True,True,True,True,True,True,True,True,True,True]
    , method="knn", params={"k":3})
show_learning_methods_performance([True,True,True,True,True,True,True,True,True,True]
    , method="dt")
show_learning_methods_performance([True,True,True,True,True,True,True,True,True,True]
    , method="rbm", params={})
show_learning_methods_performance([True,True,True,True,True,True,True,True,True,True]
    , method="nb")
show_learning_methods_performance([True,True,True,True,True,True,True,True,True,True]
    , method="svm", params={})
'''




'''
cp,pt = predict(crossVal=True, method="rf", params={"k":3}, showAllPredictions=True, featToExtract=[True,True]
                , webApp=False, plotPosAlgn=False, plotNegAlgn=False, plotTestAlgn=False, plotAIO=True)
print(Counter(cp))
'''

#create_alignments_files(aligPos=True, aligNeg=True, aligTest=False, verbose=True)


'''
#create allergens data set combining different datasets
create_fasta_file_without_duplications(['alignments/allertop+allerhunter/reduced_all_allergens.fasta'
                                           , 'alignments/allertop+allerhunter/training.allergen.fa'
                                        ]
                                       , 'alignments/allertop+allerhunter/allergens.fasta')


'''

'''
#creacion del conjunto de test todos alérgenos
create_fasta_file_without_duplications(['alignments/AllerHunter/independentdata/indp.allergen.fa'
                                           , 'alignments/AllerHunter/testingdata/testing.allergen.fa'
                                           , 'alignments/reduced_all_allergens.fasta'
                                           #, 'alignments/unitprot/cdep91allergens.fasta'
                                           , 'alignments/allerdictor/A/alg.fa'
                                           , 'alignments/allerdictor/B/alg.fa'
                                           , 'alignments/allerdictor/C/alg.fa'
                                           #, 'alignments/allerpred/testa1.fasta'
                                           #, 'alignments/allerpred/testa2.fasta'
                                           #, 'alignments/allerpred/testa3.fasta'
                                           #, 'alignments/allerpred/testa4.fasta'
                                           #, 'alignments/allerpred/testa5.fasta'
                                        ]
                                       , allTestDataset
                                       , splitFastaSeqs(allDataset)[1])

'''

'''
sid, s = splitFastaSeqs('alignments/reduced_all_allergens.fasta')
print("AllerTop total allergens: " + str(len(s)))
sid, s = splitFastaSeqs('alignments/reduced_all_nonallergens.fasta')
print("AllerTop total non allergens: " + str(len(s)))
sid, s = splitFastaSeqs('alignments/allerdictor/A/alg.fa')
print("Allerdictor A total allergens: " + str(len(s)))
sid, s = splitFastaSeqs('alignments/allerdictor/A/nlg.fa')
print("Allerdictor A total non allergens: " + str(len(s)))
sid, s = splitFastaSeqs('alignments/AllerHunter/trainingdata/training.allergen.fa')
print("AllerHunter total allergens: " + str(len(s)))
sid, s = splitFastaSeqs('alignments/AllerHunter/testingdata/testing.allergen.fa')
print("AllerHunter total test allergens: " + str(len(s)))
sid, s = splitFastaSeqs('alignments/AllerHunter/trainingdata/training.putative_non_allergen.fa')
print("AllerHunter total non allergens: " + str(len(s)))
sid, s = splitFastaSeqs('alignments/allertop+allerhunter/allergens.fasta')
print("AllerHunter + AllerTop without duplications total allergens: " + str(len(s)))

sid, s = splitFastaSeqs(allDataset)
print("Nuestro conjunto total de entrenamiento de allergens: " + str(len(s)))
sid, s = splitFastaSeqs(allTestDataset)
print("Nuestro conjunto total de test de allergens: " + str(len(s)))
d=how_many_seqs_from_a_are_duplicated_in_b(allTestDataset, allDataset)
print("duplicaciones test allergens en training allergens: "+str(len(d)))



d=how_many_seqs_from_a_are_duplicated_in_b('alignments/allertop/reduced_all_allergens.fasta'
                                           , allDataset)
print("duplicaciones allertop: "+str(len(d)))

d=how_many_seqs_from_a_are_duplicated_in_b('alignments/allerdictor/B/alg.fa'
                                           , allDataset)
print("duplicaciones allerdictor B: "+str(len(d)))

d=how_many_seqs_from_a_are_duplicated_in_b('alignments/allerdictor/C/alg.fa'
                                           , allDataset)
print("duplicaciones allerdictor C: "+str(len(d)))

d=how_many_seqs_from_a_are_duplicated_in_b('alignments/AllerHunter/testingdata/testing.allergen.fa'
                                           , allDataset)
print("duplicaciones allerhunter: "+str(len(d)))

d=how_many_seqs_from_a_are_duplicated_in_b('alignments/AllerHunter/independentdata/indp.allergen.fa'
                                           , allDataset)
print("duplicaciones allerhunter: "+str(len(d)))

d=how_many_seqs_from_a_are_duplicated_in_b('alignments/allerpred/testa1.fasta'
                                           , allDataset)
print("duplicaciones allerpred: "+str(len(d)))
d=how_many_seqs_from_a_are_duplicated_in_b('alignments/allerpred/testa2.fasta'
                                           , allDataset)
print("duplicaciones allerpred: "+str(len(d)))
d=how_many_seqs_from_a_are_duplicated_in_b('alignments/allerpred/testa3.fasta'
                                           , allDataset)
print("duplicaciones allerpred: "+str(len(d)))
d=how_many_seqs_from_a_are_duplicated_in_b('alignments/allerpred/testa4.fasta'
                                           , allDataset)
print("duplicaciones allerpred: "+str(len(d)))
d=how_many_seqs_from_a_are_duplicated_in_b('alignments/allerpred/testa5.fasta'
                                           , allDataset)
print("duplicaciones allerpred: "+str(len(d)))

print("duplicaciones allerpred: "+str(len(d)))
d=how_many_seqs_from_a_are_duplicated_in_b('alignments/unitprot/cdep91allergens.fasta'
                                           , allDataset)
print("duplicaciones unitprot: "+str(len(d)))

'''

#show_learning_methods_performance([True,True], method="rf")

#show_learning_methods_performance([True,True], method="rf")

#show_all_learning_methods_performance(featToExtract=[True,True])
#show_all_learning_methods_performance(featToExtract=[True,True,True,True,True,True,True,True,True,True])

#show_learning_methods_performance(method="nb", m=1, nExp= 10, featToExtract=[True,True,True,True,True,True,True,True,True,True])

#show_all_learning_methods_performance()
#show_all_learning_methods_performance(methods=[("mlp",{}), ("rf", {}), ("knn", {"k":3}), ("dt", {}), ("nb", {})], [True,True,True,True,True,True,True,True,True,True])
#show_learning_methods_performance(method="mlp", featToExtract=[True,True,True,True,True,True,True,True,True,True])

