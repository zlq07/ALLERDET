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
Fecha: 30-06-2016 (ultima modificacion: 26/08/2022)
----------------------------------------------------------
"""
from collections import Counter
import numpy as np

try:
    from app.predict import predict
    from app.predict import prediction_test
    from app.predict import tuning_model
    from app.predict import tuning_model_performance
    from app.predict import show_learning_methods_performance
    from app.predict import tpr_prod_tnr_score
    from app.alignment import splitFastaSeqs
    from app.alignment import how_many_seqs_from_a_are_duplicated_in_b
    from app.alignment import create_fasta_file_without_duplications
    from app.alignment import create_alignments_files
    from app.alignment import secuences_by_ids
except SystemError:
    from predict import predict
    from predict import tuning_model
    from predict import show_learning_methods_performance
    from alignment import splitFastaSeqs
    from alignment import how_many_seqs_from_a_are_duplicated_in_b
    from alignment import create_fasta_file_without_duplications
    from alignment import create_alignments_files
    from alignment import secuences_by_ids


if __name__ == '__main__':

    ##----------------------------------------------------------
    ## Data set
    ##----------------------------------------------------------
    aAllergenonline = 'app/alignments/allergenonline/allergenonline.fasta'
    aAllerTop = 'app/alignments/allertop/reduced_all_allergens.fasta'
    naAllerTop = 'app/alignments/allertop/reduced_all_nonallergens.fasta'
    # aAllerdictorA = 'app/alignments/allerdictor/A/alg.fa'
    # naAllerdictorA = 'app/alignments/allerdictor/A/nlg.fa'
    # aAllerdictorB = 'app/alignments/allerdictor/B/alg.fa'
    # aAllerdictorC = 'app/alignments/allerdictor/C/alg.fa'
    aAllerHunter = 'app/alignments/AllerHunter/trainingdata/training.allergen.fa'
    naAllerHunter = 'app/alignments/AllerHunter/trainingdata/training.putative_non_allergen.fa'
    aAllerHunterTest = 'app/alignments/AllerHunter/testingdata/testing.allergen.fa'
    aAllerHunterInd = 'app/alignments/AllerHunter/independentdata/indp.allergen.fa'
    aAllerPred1 = 'app/alignments/allerpred/testa1.fasta'
    aAllerPred2 = 'app/alignments/allerpred/testa2.fasta'
    aAllerPred3 = 'app/alignments/allerpred/testa3.fasta'
    aAllerPred4 = 'app/alignments/allerpred/testa4.fasta'
    aAllerPred5 = 'app/alignments/allerpred/testa5.fasta'
    naUniprot = 'app/alignments/unitprot/nonallergens-uniprot.fasta'
    aUnitprot2022 = 'app/alignments/unitprot/uniprot-allergen2022.fasta'
    aUnitprot2022andHypers = 'app/alignments/unitprot/uniprot-allergen2022_including_hypersensitive.fasta'
    aCOMPARE = 'app/alignments/COMPARE/COMPARE-2022-FastA-Seq.fasta'

    myAllergenDataset = 'app/alignments/allergens_data_set.fasta'
    myNonAllergenDataset = 'app/alignments/nonallergens_data_set.fasta'
    myCreatedFromWebTestDataset = 'app/alignments/created_test.fasta'
    myValidationDataset = 'app/alignments/validation_allergens.fasta'
    myTestDataset = 'app/alignments/test_allergens_data_set.fasta'
    myNonAllergenTestDataset = 'app/alignments/test_nonallergens_data_set.fasta'

    #Nuestro Data set
    sid, s = splitFastaSeqs(myAllergenDataset)
    print("Nuestro conjunto total de entrenamiento de allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(myNonAllergenDataset)
    print("Nuestro conjunto total de entrenamiento de non-allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(myValidationDataset)
    print("Conjunto total de validación de allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(myTestDataset)
    print("Nuestro conjunto total de test de allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(myNonAllergenTestDataset)
    print("Nuestro conjunto total de test de non-allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(myCreatedFromWebTestDataset)
    print("Conjunto total de test consultado en web: " + str(len(s)))
    print("------")

    #resto de datasets disponibles
    sid, s = splitFastaSeqs(aAllergenonline)
    print("Allergenonline total allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerTop)
    print("AllerTop total allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerdictorA)
    print("Allerdictor A total allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerdictorB)
    print("Allerdictor B total allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerdictorC)
    print("Allerdictor C total allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerHunter)
    print("AllerHunter total allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerHunterTest)
    print("AllerHunter total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerHunterInd)
    print("AllerHunter total indep allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerPred1)
    print("AllerPred 1 total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerPred2)
    print("AllerPred 2 total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerPred3)
    print("AllerPred 3 total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerPred4)
    print("AllerPred 4 total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aAllerPred5)
    print("AllerPred 5 total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aUnitprotReviews)
    print("Uniprot total reviewed allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aUnitprot)
    print("Uniprot total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aUnitprot2)
    print("Uniprot2 total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aUnitprot2022)
    print("Uniprot2022 total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aCOMPARE)
    print("COMPARE total allergens: " + str(len(s)))
    print("------")
    sid, s = splitFastaSeqs(naAllerTop)
    print("AllerTop total non-allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(naAllerHunter)
    print("AllerHunter total non-allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(naAllerdictorA)
    print("Allerdictor A total non-allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(naUniprot)
    print("Uniprot A total non-allergens: " + str(len(s)))
    print("------")

    #duplicaciones
    d = how_many_seqs_from_a_are_duplicated_in_b(aCOMPARE, aUnitprot2022)
    print("duplicaciones COMPARE y UnitProt2022: " + str(len(d)))
    # d=how_many_seqs_from_a_are_duplicated_in_b(myValidationDataset, myAllergenDataset)
    # print("duplicaciones validation en train: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(myTestDataset, myAllergenDataset)
    print("duplicaciones test en train: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerTop, myAllergenDataset)
    print("duplicaciones allertop: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerdictorA, myAllergenDataset)
    print("duplicaciones allerdictor A: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerdictorB, myAllergenDataset)
    print("duplicaciones allerdictor B: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerdictorC, myAllergenDataset)
    print("duplicaciones allerdictor C: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerHunter, myAllergenDataset)
    print("duplicaciones allerhunter train: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerHunterTest, myAllergenDataset)
    print("duplicaciones allerhunter test: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerHunterInd, myAllergenDataset)
    print("duplicaciones allerhunter indep: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerPred1, myAllergenDataset)
    print("duplicaciones allerpred 1: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerPred2, myAllergenDataset)
    print("duplicaciones allerpred 2: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerPred3, myAllergenDataset)
    print("duplicaciones allerpred 3: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerPred4, myAllergenDataset)
    print("duplicaciones allerpred 4: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aAllerPred5, myAllergenDataset)
    print("duplicaciones allerpred 5: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aUnitprot, myAllergenDataset)
    print("duplicaciones unitprot: "+str(len(d)))
    d=how_many_seqs_from_a_are_duplicated_in_b(aUnitprot2, myAllergenDataset)
    print("duplicaciones unitprot2: "+str(len(d)))
    print("------")
    # d=how_many_seqs_from_a_are_duplicated_in_b(naAllerTop, myNonAllergenDataset)
    # print("duplicaciones allertop: "+str(len(d)))
    # d=how_many_seqs_from_a_are_duplicated_in_b(naAllerHunter, myNonAllergenDataset)
    # print("duplicaciones allerhunter: "+str(len(d)))
    # d=how_many_seqs_from_a_are_duplicated_in_b(naAllerdictorA, myNonAllergenDataset)
    # print("duplicaciones allerdictor: "+str(len(d)))
    # d=how_many_seqs_from_a_are_duplicated_in_b(naUniprot, myNonAllergenDataset)
    # print("duplicaciones uniprot: "+str(len(d)))
    print()
    # d = how_many_seqs_from_a_are_duplicated_in_b(myValidationDataset, myTestDataset)
    # print("duplicaciones validacion y test: " + str(len(d)))
    # d = how_many_seqs_from_a_are_duplicated_in_b(myValidationDataset, myAllergenDataset)
    # print("duplicaciones validacion y allergen set: " + str(len(d)))
    d = how_many_seqs_from_a_are_duplicated_in_b(myAllergenDataset, myTestDataset)
    print("duplicaciones allergen set entre train y test: " + str(len(d)))
    d = how_many_seqs_from_a_are_duplicated_in_b(myNonAllergenDataset, myNonAllergenTestDataset)
    print("duplicaciones non allergen set entre train y test: " + str(len(d)))
    print()

    ##----------------------------------------------------------------------------------------
    ## Crear Datasets
    ##----------------------------------------------------------------------------------------
    #keep 1000 allergen sequences from allertop
    # create_fasta_file_without_duplications([aAllerTop]
    #                                              , 'app/alignments/allergens_allertop_1000.fasta'
    #                                              , maxSec=1000)

    # keep 500 allergen sequences of previous file from allertop
    # create_fasta_file_without_duplications(['app/alignments/allergens_allertop_1000.fasta']
    #                                        , 'app/alignments/allergens_allertop_500_1000.fasta'
    #                                        , maxSec=500)

    #keep 2500 non-allergen sequences from allerhunter
    # create_fasta_file_without_duplications([naAllerHunter]
    #                                              , 'app/alignments/nonallergens_allerhunter_2500.fasta', maxSec=2500)

    #
    # # combina diferentes datasets de alérgenos sin duplicaciones
    # create_fasta_file_without_duplications(['alignments/allergens_fn_allertop.fasta.fasta'
    #                                         , aUnitprotReviews]
    #                                             #aAllerTop, aAllerHunter, aUnitprot2, aAllerdictorA]
    #                                              , 'alignments/allergens_data_set.fasta'
    #                                              # , splitFastaSeqs('alignments/allergens_allertop_1000.fasta')[1]
    #                                              )


    # combina diferentes datasets de alérgenos sin duplicaciones
    # our allergen dataset
    create_fasta_file_without_duplications([
                                            aUnitprot2022,
        # aUnitprot2022andHypers,
        aAllergenonline
        ,aCOMPARE,
        # aAllerPred1,aAllerPred2,aAllerPred3,aAllerPred4,aAllerPred5,
        # aAllerTop,
        # aAllerHunter,
        # aAllerHunterInd,
        # aAllerHunterTest

    ]
                                                #aAllerTop, aAllerHunter, aUnitprot2, aAllerdictorA]
                                                 , myAllergenDataset
                                                    #exclude the following:
                                                 #, splitFastaSeqs('app/alignments/allergens_allertop_1000.fasta')[1]
                                                , maxSec=80.0
                                                # , maxSec=20000
                                                , shuffle=True
                                                 )
    sid, s = splitFastaSeqs(myAllergenDataset)
    print("Our allergen train dataset: " + str(len(s)))


    # dataset non allergens
    create_fasta_file_without_duplications(['app/alignments/unitprot/non/plant_nonallergen.fasta'
                                               , 'app/alignments/unitprot/non/cowmilk_nonallergen.fasta'
                                               , 'app/alignments/unitprot/non/eggs_nonallergen.fasta'
                                               , 'app/alignments/unitprot/non/salmo-nonallergen.fasta'
                                                , naAllerTop, naAllerHunter
                                            ]
                                           , myNonAllergenDataset, maxSec=3790
                                           , shuffle=True
                                           )
    sid, s = splitFastaSeqs(myNonAllergenDataset)
    print("Our non-allergen train dataset: " + str(len(s)))



    # create allergen dataset for testing from different sources
    exclusion=splitFastaSeqs(myAllergenDataset)[1]
    # exclusion.extend(splitFastaSeqs(myValidationDataset)[1])
    # exclusion.extend(splitFastaSeqs(myTestDataset)[1])
    create_fasta_file_without_duplications([
                                            aUnitprot2022
                                                # aUnitprot2022andHypers
                                                , aAllergenonline
                                                , aCOMPARE
                                                # , aAllerPred1,aAllerPred2,aAllerPred3,aAllerPred4,aAllerPred5
                                                # , aAllerTop
                                                # , aAllerHunter
                                                # , aAllerHunterInd
                                                # , aAllerHunterTest
                                            ]
                                                 , myTestDataset, exclusion
                                                , maxSec=100. #in reality, is 20.0%
                                                , shuffle=True
    )
    sid, s = splitFastaSeqs(myTestDataset)
    print("Our Test allergens: " + str(len(s)))


    # creacion del conjunto de test de no alérgenos
    exclusion=splitFastaSeqs(myNonAllergenDataset)[1]
    create_fasta_file_without_duplications(['app/alignments/unitprot/non/plant_nonallergen.fasta'
                                               , 'app/alignments/unitprot/non/cowmilk_nonallergen.fasta'
                                               , 'app/alignments/unitprot/non/eggs_nonallergen.fasta'
                                               , 'app/alignments/unitprot/non/salmo-nonallergen.fasta'
                                                , naAllerTop, naAllerHunter
                                                    ]
                                                 , myNonAllergenTestDataset, exclusion, 948)
    sid, s = splitFastaSeqs(myNonAllergenTestDataset)
    print("Our non-allergen test dataset: " + str(len(s)))


    #possible crash. It is better to execute the command directly from cmd in the folder alignments/
    # create_alignments_files(aligPos=True, aligNeg=True, aligTest=True, testSecFile='test_allergens_data_set.fasta', verbose=True)
    # create_alignments_files(posSecFile="allergens_data_set.fasta", negSecFile="nonallergens_data_set.fasta")


    ##----------------------------------------------------------------------------------------
    ## Performance predicciones
    ##----------------------------------------------------------------------------------------
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score


    tuning_model_performance("dt", maxM=1, verbose=True, score={'tpr_prod_tnr':make_scorer(tpr_prod_tnr_score), 'accuracy': make_scorer(accuracy_score)}, refit='recall')

    tuning_model_performance("dt", maxM=1, verbose=True, score={'recall':make_scorer(recall_score),'accuracy':make_scorer(accuracy_score)}, refit='recall')
    tuning_model_performance("nb", maxM=1, verbose=True)
    tuning_model_performance("knn", maxM=1, verbose=True)
    tuning_model_performance("mlp", maxM=1, verbose=True)
    tuning_model_performance("rbm", maxM=1, reduction=100,  verbose=True, score={'recall':make_scorer(recall_score),'accuracy':make_scorer(accuracy_score)}, refit='recall')

    cp, pt = predict(method="rbm", params={
        'rbm__n_iter': 20, 'rbm__n_components': 50, 'rbm__learning_rate': 0.1
        , "mod": "dt"
        , "mod_par": {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 50}}
                     , m=1
                     , showAllPredictions=True, featToExtract=[True], webApp=False
                     , testAlFile="a_cdp.txt" #"a_cdp_review.txt"
                     , printNativeClassReport= True
                     , plotPosAlgn=False
                     , plotNegAlgn=False
                     , plotAIO=False
                     , kFolds=0)

    counter = Counter(cp)
    print(str(counter))
    print("Accuracy: " + str((counter[1] * 100) / len(cp)))


    #Pruebas de predicción de alérgenos


    print("Prueba Alérgenos DT")
    cp,pt = predict(method="dt", params={'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 8}
                    , m=1, featToExtract=[True, False, True]
                    , showAllPredictions=True, webApp=False
                    , printNativeClassReport=True
                    , testAlFile="a_cdp.txt")#"a_cdp_review.txt")
    counter = Counter(cp)
    print(str(counter))
    print("Accuracy: " + str((counter[1]*100)/len(cp)))

    # print("Prueba Alérgenos RF")
    # cp,pt = predict(method="rf", params={}
    #                 , showAllPredictions=False, featToExtract=[True,True], webApp=False)
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter['allergen']*100)/len(cp)))
    #
    # print("Prueba Alérgenos NB")
    # cp,pt = predict(method="nb", params={'priors': [0.5, 0.5]}
    #                 , showAllPredictions=False, featToExtract=[True,True], webApp=False)
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter['allergen']*100)/len(cp)))
    #
    print("Prueba Alérgenos RBM")
    cp,pt = predict(method="rbm", params={'rbm__n_iter': 20, 'rbm__n_components': 1000, 'rbm__learning_rate': 0.001}
                    , showAllPredictions=False, featToExtract=[True,True, True, True], webApp=False)
    counter = Counter(cp)
    print(str(counter))
    print("Accuracy: " + str((counter['allergen']*100)/len(cp)))

    # print("Prueba Alérgenos MLP")
    # cp,pt = predict(method="mlp", params={'solver': 'lbfgs', 'max_iter': 300, 'hidden_layer_sizes': (100,)
    #     , 'activation': 'logistic'}, showAllPredictions=False, featToExtract=[True,True], webApp=False)
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter['allergen']*100)/len(cp)))
    #
    # print("Prueba Alérgenos k-medias")
    # cp,pt = predict(method="km", params={'max_iter': 300, 'n_clusters': 2, 'algorithm': 'full'}
    #                 , showAllPredictions=False, featToExtract=[True,True], webApp=False)
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter[0]*100)/len(cp)))
    #
    # # Pruebas de predicción de no alérgenos
    #
    print("Prueba NO Alérgenos kNN")
    cp, pt = predict(method="knn", params={'metric': 'euclidean', 'weights': 'uniform'
        , 'algorithm': 'auto', 'leaf_size': 1, 'p': 1, 'n_neighbors': 3}, showAllPredictions=False,
                     featToExtract=[True, True, True, True]
                     , webApp=False, plotPosAlgn=False, plotNegAlgn=False, plotTestAlgn=False, plotAIO=False,
                     figSameColor=False,testAlFile="a_cdpn.txt")
    counter = Counter(cp)
    print(str(counter))
    print("Accuracy: " + str((counter['non-allergen'] * 100) / len(cp)))

    # print("Prueba NO Alérgenos DT")
    # cp, pt = predict(method="dt", params={'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 8}
    #                  , showAllPredictions=False, featToExtract=[True, True], webApp=False,testAlFile="a_cdpn.txt")
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter['non-allergen'] * 100) / len(cp)))
    #
    # print("Prueba NO Alérgenos RF")
    # cp, pt = predict(method="rf", params={}
    #                  , showAllPredictions=False, featToExtract=[True, True], webApp=False,testAlFile="a_cdpn.txt")
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter['non-allergen'] * 100) / len(cp)))
    #
    # print("Prueba NO Alérgenos NB")
    # cp, pt = predict(method="nb", params={'priors': [0.5, 0.5]}
    #                  , showAllPredictions=False, featToExtract=[True, True], webApp=False,testAlFile="a_cdpn.txt")
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter['non-allergen'] * 100) / len(cp)))
    #
    print("Prueba NO Alérgenos RBM")
    cp, pt = predict(method="rbm", params={'rbm__n_iter': 20, 'rbm__n_components': 1000, 'rbm__learning_rate': 0.001}
                     , showAllPredictions=False, featToExtract=[True, True, True, True]
        , webApp=False,testAlFile="a_cdpn.txt")
    counter = Counter(cp)
    print(str(counter))
    print("Accuracy: " + str((counter['non-allergen'] * 100) / len(cp)))

    # print("Prueba NO Alérgenos MLP")
    # cp, pt = predict(method="mlp", params={'solver': 'lbfgs', 'max_iter': 300, 'hidden_layer_sizes': (100,)
    #     , 'activation': 'logistic'}, showAllPredictions=False, featToExtract=[True, True]
    #     , webApp=False,testAlFile="a_cdpn.txt")
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter['non-allergen'] * 100) / len(cp)))
    #
    # print("Prueba NO Alérgenos k-medias")
    # cp, pt = predict(method="km", params={'max_iter': 300, 'n_clusters': 2, 'algorithm': 'full'}
    #                  , showAllPredictions=False, featToExtract=[True, True], webApp=False,testAlFile="a_cdpn.txt")
    # counter = Counter(cp)
    # print(str(counter))
    # print("Accuracy: " + str((counter[0] * 100) / len(cp)))



    # ids = [pt[i][1] for i, c in enumerate(cp) if c == 'non-allergen' and len(pt[i]) > 1]
    # print(len(ids))
    # # print(ids[:3])
    # # # # creacion del nuevo conjunto de test de alérgenos
    # exclusion = splitFastaSeqs(myAllergenDataset)[1]
    # ex=secuences_by_ids(myTestDataset, ids)
    # print(len(ex))
    # exclusion.extend(ex)
    # create_fasta_file_without_duplications([myTestDataset, myValidationDataset],myTestDataset, exclusion)


    # accuracy con cross validation k-fold=10 y estratificación

    show_learning_methods_performance([True,True, True, True], method="knn", params={'metric': 'euclidean', 'weights': 'uniform'
        , 'algorithm': 'auto', 'leaf_size': 1, 'p': 1, 'n_neighbors': 3}, kFolds=10)

    # show_learning_methods_performance([True,True], method="rf", params={}, kFolds=10)

    # show_learning_methods_performance([True,True], method="dt"
    #                                   , params={'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 8}, kFolds=10)
    #
    # show_learning_methods_performance([True,True], method="nb", params={'priors': [0.5, 0.5]}, kFolds=10)
    #
    # show_learning_methods_performance([True,True], method="mlp", params={'solver': 'lbfgs', 'max_iter': 300
    #     , 'hidden_layer_sizes': (100,), 'activation': 'logistic'}, kFolds=10)
    # #
    show_learning_methods_performance([True,True, True, True], method="rbm", params={'rbm__n_iter': 20, 'rbm__n_components': 1000
        , 'rbm__learning_rate': 0.001}, kFolds=10)

































