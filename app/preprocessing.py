#!venv/bin/python3
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
Fecha: 30-06-2016 (ultima modificacion: 21/05/2017)
----------------------------------------------------------
"""

import os

def extract_all_features_and_classifications(featToExtract=[True, True], m=1
                                             , posAlFile="a_cdep.txt", negAlFile="a_cden.txt", testAlFile="a_cdp.txt"
                                             , testClass="allergen", appFolder="/app/", alignsPath="/alignments/"
                                             , verbose=False):
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

    if os.path.exists(workingDir + alignsPath):
        path = workingDir + alignsPath
    else:
        path = alignsPath

    f1, f2, c1, c2, ft, ct, p1, p2, pi, pit = [], [], [], [], [], [], [], [], [], []

    # características y clasificaciones del conjunto de entrenamiento "allergen"
    if posAlFile != "":
        f1, c1, p1 = get_features(path + posAlFile, m, "allergen", featToExtract)

    # características y clasificaciones del conjunto de entrenamiento "non-allergen"
    if negAlFile != "":
        f2, c2, p2 = get_features(path + negAlFile, m, "non-allergen", featToExtract)

    # caracteristicas y clasificaciones del conjunto de prueba
    if testAlFile != "":
        ft, ct, pit = get_features(path + testAlFile, m, testClass, featToExtract)

    f = f1 + f2  # caracteristicas del conjunto de entrenamiento
    c = c1 + c2  # clasificaciones del conjunto de entrenamiento
    pi = p1 + p2

    # eliminar elementos repetidos del conjunto de entrenamiento
    d = []
    for i, x in enumerate(f):
        if x in d:
            f.pop(i)
            c.pop(i)
            pi.pop(i)

        else:
            d.append(x)
    if verbose:
        print("train " + str(len(f)) + ", test " + str(len(ft)))
    return f, c, ft, ct, pi, pit


def get_features(filename, m, classType, featToExtract=[True, True], verbose=False):
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

    if os.path.exists(workingDir + filename):
        filename = workingDir + filename

    content = ""
    with open(filename, "rt") as f:
        for line in f:
            content+=line

        if verbose:
            print("file "+filename+" read")

        # f = open(filename, 'rt')
        # content = f.read()
        presec = content.split(">>>")[1:]  # excluimos la parte inicial del fichero
        aligmentsNumbers = []
        features = []
        classif = []
        protInfos = []

        for alignment in presec:
            descp_line = alignment.split("\n")[0]
            protInfo = parse_info_protein(descp_line)

            # extracción de características
            ali = alignment.split(">>")  # ("Smith-Waterman score:")
            if len(ali) > 1:
                ali = ali[1:]  # quitamos el resumen de los mejores alineamientos
                aBest = []

                for a in ali:
                    firstLine = a.split("\n")[0]
                    # sólo las 2 líneas del alineamiento donde están las características
                    feats = " ".join(a.split("\n")[1:3])  # (";") #excluyendo la primera (nombre de la proteina)
                    s = feats.split("Smith-Waterman score:")
                    score = int(s[1].split(";")[0].strip())

                    lengthStr = firstLine.split("aa)")[0].split("(")
                    length = lengthStr[1].strip()

                    if not length.isdigit():
                        length = lengthStr[len(lengthStr) - 1].split("(")[0].strip();
                        if not length.isdigit():
                            length = length.split("a`)")[0]

                    length = int(length)  # int(s[1].split(") in ")[1].split("aa")[0].strip())

                    identity = 0
                    totalExtractions = len(featToExtract)

                    # if totalExtractions > 2:
                    extractions = []
                    extIdentity = totalExtractions >= 3 and featToExtract[2]  # condición extraer identity
                    if extIdentity:
                        identity = float(s[1].split("% identity")[0].split(";")[1].strip())
                    if not extIdentity or (extIdentity and identity >= 35):  # FAO/WHO guidelines (identity >= 35%)
                        extractions = []
                        for i in range(len(featToExtract)):
                            extract = featToExtract[i]

                            if extract:
                                if i == 0:  # Smith-Waterman Score
                                    feat = score
                                elif i == 1:  # Alignment Length
                                    feat = length
                                elif i == 2:  # % Identity
                                    feat = identity
                                elif i == 3:  # % Similarity
                                    feat = float(s[1].split("% similar")[0].split("% identity (")[1].strip())
                                elif i == 4:  # Z-Score
                                    feat = float(s[0].split("Z-score:")[1].split("bits")[0].strip())
                                elif i == 5:  # initn
                                    feat = int(s[0].split("initn:")[1].split("init1")[0].strip())
                                elif i == 6:  # init1
                                    feat = int(s[0].split("init1:")[1].split("opt")[0].strip())
                                elif i == 7:  # bits
                                    feat = float(s[0].split("bits:")[1].split("E(")[0].strip())
                                elif i == 8:  # E value
                                    feat = float(s[0].split("E(")[1].split("):")[1].strip())

                                if feat != None:
                                    extractions.append(feat)
                    if extractions != []:
                        aBest.append(extractions)
                    # else:
                    #     aBest.append([score, length])

                aBest.sort(reverse=True)  # ordenamos de mayor a menor score
                bestMs = aBest[:m] # los mejores m alineamientos según el primer elemento
                features.extend(bestMs)
                protInfos.extend([protInfo] * len(bestMs)) #duplicamos el nombre de la proteína segun los bestMs

        classif = [classType] * len(features) #generamos el vector de clasificación según el número de features

    return features, classif, protInfos

def parse_info_protein(descp_line):
    '''

    :param descp_line: línea de descripción de la proteína
    :return: lista con mínimo una componente (el nombre de la proteína) o de tres componentes
    (el identificador de la base de datos, el identificador de la proteína en dicha base de datos, y el nombre de la
    proteína) en ese orden.
    '''
    pro = descp_line.split("|")  # descripción de la proteína
    protName = pro[0]

    # nombres de las proteínas
    if pro[0] == "gi":
        protName = pro[1] if len(pro) > 1  else ""
        protName = pro[3] if len(pro) > 3  else protName
        protName = pro[4] if len(pro) > 4 else protName

    elif pro[0] == "sp" or pro[0] == "tr":
        p = pro[2].split(" ")
        protName = " ".join(p[1:]).split("PE")[0] #si el archivo es del resultado de un alineamiento
        if protName == "": #si el archivo es un FASTA normal
            protName = " ".join(p[1:])

    if len(pro) > 1:
        protInfo = [pro[0], pro[1], protName]
    else:
        protInfo = [protName]
    return protInfo

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
    res = []
    for i in range(max(min(9, len(featToExtract)), 1)):  # max de 9 features
        extract = featToExtract[i]

        if extract:
            if i == 0:  # Smith-Waterman Score
                feat = "Smith-Waterman Score"
            elif i == 1:  # Alignment Length
                feat = "Alignment Length"
            elif i == 2:  # % Identity
                feat = "Identity"
            elif i == 3:  # % Similarity
                feat = "Similarity"
            elif i == 4:  # Z-Score
                feat = "Z-Score"
            elif i == 5:  # initn
                feat = "initn"
            elif i == 6:  # init1
                feat = "init1"
            elif i == 7:  # bits
                feat = "bits"
            elif i == 8:  # E value
                feat = "E value"
            res.append(feat)
    return ",".join(res)