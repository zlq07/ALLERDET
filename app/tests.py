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
Fecha: 30-06-2016 (ultima modificacion: 21/05/2017)
----------------------------------------------------------
"""

from collections import Counter
try:
    from .predict import predict
    from .predict import tuning_model
    from .predict import show_learning_methods_performance
    from .predict import tuning_model_performance
    from .predict import prediction_test
    from .alignment import splitFastaSeqs
    from .alignment import how_many_seqs_from_a_are_duplicated_in_b
    from .alignment import create_fasta_file_without_duplications
    from .alignment import create_alignments_files
    from .alignment import secuences_by_ids
    from .preprocessing import features_for_extracting_to_string
except SystemError:
    from predict import predict
    from predict import tuning_model
    from predict import show_learning_methods_performance
    from alignment import splitFastaSeqs
    from alignment import how_many_seqs_from_a_are_duplicated_in_b
    from alignment import create_fasta_file_without_duplications
    from alignment import create_alignments_files
    from alignment import secuences_by_ids
    from preprocessing import features_for_extracting_to_string
    from predict import tuning_model_performance
    from predict import prediction_test

if __name__ == '__main__':


    ##----------------------------------------------------------
    ## Data set
    ##----------------------------------------------------------
    aAllerTop = 'alignments/reduced_all_allergens.fasta'
    naAllerTop = 'alignments/reduced_all_nonallergens.fasta'
    aAllerdictorA = 'alignments/allerdictor/A/alg.fa'
    naAllerdictorA = 'alignments/allerdictor/A/nlg.fa'
    aAllerdictorB = 'alignments/allerdictor/B/alg.fa'
    aAllerdictorC = 'alignments/allerdictor/C/alg.fa'
    aAllerHunter = 'alignments/AllerHunter/trainingdata/training.allergen.fa'
    naAllerHunter = 'alignments/AllerHunter/trainingdata/training.putative_non_allergen.fa'
    aAllerHunterTest = 'alignments/AllerHunter/testingdata/testing.allergen.fa'
    aAllerHunterInd = 'alignments/AllerHunter/independentdata/indp.allergen.fa'
    aAllerPred1 = 'alignments/allerpred/testa1.fasta'
    aAllerPred2 = 'alignments/allerpred/testa2.fasta'
    aAllerPred3 = 'alignments/allerpred/testa3.fasta'
    aAllerPred4 = 'alignments/allerpred/testa4.fasta'
    aAllerPred5 = 'alignments/allerpred/testa5.fasta'
    aUnitprot = 'alignments/unitprot/cdep91allergens.fasta'
    aUnitprot2 = 'alignments/unitprot/uniprot-allergen.fasta'
    naUniprot = 'alignments/unitprot/nonallergens-uniprot.fasta'
    aUnitprotReviews = 'alignments/unitprot/uniprot-allergy+OR+atopy+OR+allergen+OR+allergome.fasta'

    myAllergenDataset = 'alignments/allergens_data_set.fasta'
    myNonAllergenDataset = 'alignments/nonallergens_data_set.fasta'
    myCreatedFromWebTestDataset = 'alignments/created_test.fasta'
    myValidationDataset = 'alignments/validation_allergens.fasta'
    myTestDataset = 'alignments/test_allergens_data_set.fasta'
    myNonAllergenTestDataset = 'alignments/test_nonallergens_data_set.fasta'

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
    print()

    #resto de datasets disponibles
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
    sid, s = splitFastaSeqs(aUnitprot)
    print("Uniprot total test allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(aUnitprot2)
    print("Uniprot2 total test allergens: " + str(len(s)))
    print("------")
    sid, s = splitFastaSeqs(naAllerTop)
    print("AllerTop total non-allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(naAllerHunter)
    print("AllerHunter total non-allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(naAllerdictorA)
    print("Allerdictor A total non-allergens: " + str(len(s)))
    sid, s = splitFastaSeqs(naUniprot)
    print("Uniprot A total non-allergens: " + str(len(s)))
    print()

    #duplicaciones
    d=how_many_seqs_from_a_are_duplicated_in_b(myValidationDataset, myAllergenDataset)
    print("duplicaciones validation en train: "+str(len(d)))
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
    print()
    d = how_many_seqs_from_a_are_duplicated_in_b(myAllergenDataset, myTestDataset)
    print("duplicaciones allergen set entre train y test: " + str(len(d)))
    d = how_many_seqs_from_a_are_duplicated_in_b(myNonAllergenDataset, myNonAllergenTestDataset)
    print("duplicaciones non allergen set entre train y test: " + str(len(d)))
    print()


    ##----------------------------------------------------------
    ## Predicciones
    ##----------------------------------------------------------

    cp, pt = predict(method="rbm", params={'rbm__n_iter': 20, 'rbm__n_components': 1000, 'rbm__learning_rate': 0.001
            , "mod": "dt"
            , "mod_par": {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 3}}, m=2
                     , showAllPredictions=False, featToExtract=[True, True, True], webApp=False,testAlFile="a_cdp.txt"
                     , plotPosAlgn=True, plotNegAlgn=True, plotAIO=True)
    counter = Counter(cp)
    print(str(counter))
    print("Accuracy: " + str((counter['allergen'] * 100) / len(cp)))

    prediction_test("rbm", {'rbm__n_iter': 20, 'rbm__n_components': 1000, 'rbm__learning_rate': 0.001
            , "mod": "dt"
            , "mod_par": {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 3}}, 2, [True, True, True])

    ##----------------------------------------------------------
    ## Performance
    ##----------------------------------------------------------
    tuning_model_performance("dt", verbose=True)
    tuning_model_performance("nb", verbose=True)
    tuning_model_performance("knn", maxM=1, verbose=True)
    tuning_model_performance("mlp", maxM=1, verbose=True)
    tuning_model_performance("rbm", [[True, True], [True,False, True], [True, True, True], [False, False, True, True]
                                     , [True, True, True, True]], None, maxM=1, verbose=True, reduction=100)































