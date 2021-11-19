#!/usr/bin/env python3
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
import os.path
from collections import Counter

try:
    from .preprocessing import parse_info_protein
except SystemError:
    from preprocessing import parse_info_protein

def create_alignments_files(aligPos=True, aligNeg=True, aligTest=False
                            , posSecFile="allergens_data_set.fasta"
                            , posAlFile="a_cdep.txt"
                            , negSecFile="nonallergens_data_set.fasta"
                            , negAlFile="a_cden.txt"
                            , testSecFile="created_test.fasta"
                            , testAlFile="a_cdp.txt"
                            , alignsPath="/alignments/"
                            , fastaProgramPath="fasta-36.3.8d/bin/fasta36"
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
        fastaProgramPath=alignsPath+fastaProgramPath
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

def splitFastaSeqs(filename, id_is_complete_decp_line=True):
    '''
    Extrae de un archivo FASTA sus secuencias
    :param filename: el path del archivo FASTA
    :param id_is_complete_decp_line True si el id de la secuencia va a ser la línea completa de la descripción.
    False si el id será el campo adecuado de la línea de descripción de la proteína del archivo FASTA
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
                id = m[1]
                if not id_is_complete_decp_line:
                    prot_info = parse_info_protein(m[1])
                    if len(prot_info) > 1:
                        id=prot_info[1]
                seqIds.append(id)

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

def create_fasta_file_without_duplications(filenames, resultFilename='alignments/result.fasta', seqsNotIn=[], maxSec=5000):
    '''
    Crea un fichero con extension .fasta sin incluir duplicaciones de secuencias a partir de las secuencias
    extraidas de los archivos FASTA proporcionados.
    :param filenames: una lista de paths de los archivos FASTA de los que se desea extraer sus secuencias
    :param resultFilename: el path donde se guardará el archivo final resultante
    :param seqsNotIn: (opcional) lista de secuencias que no se quieran duplicar
    :param maxSec: (opcional) número máximo de secuencias que tendrá el archivo fasta resultante
    '''
    res=[]
    for f in filenames:
        sid, seqs = splitFastaSeqs(f)
        c = Counter(seqs)

        for s in c.keys():
            i = seqs.index(s)
            rs = [sid[i], s]
            if not any(s==x[1] for x in res) and s not in seqsNotIn and rs not in res:
                res.append(rs)
                if len(res) > maxSec:
                    break
        if len(res) > maxSec:
            break

    #push the content
    content = ""
    for i,r in enumerate(res):
        if i>0:
            content += "\n"
        content += ">" + r[0] + "\n" + r[1]
    with open(resultFilename, "w") as f:
        f.write(content)

def secuences_by_ids(secs_filename, ids):
    '''
    Busca las secuencias que coincidan con los ids indicados y las devuelve
    :param secs_filename: el archivo de las secuencias
    :param ids: ids de secuencias a buscar
    :return: lista de secuencias que coinciden con los ids de secuencias indicados
    '''
    i,s = splitFastaSeqs(secs_filename, False)
    secs = list(zip(i,s))
    res=[]

    for id in ids:
        for a,b in secs:

            if id==a:
                res.append(b)
                break
    return res