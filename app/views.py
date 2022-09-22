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

from flask import Markup,render_template, flash, request
from app import app
from .forms import AminoacidSequencesForm
from .predict import predict


# @app.errorhandler(404)
# def page_not_found(e):
#     return e

@app.route('/', methods=['GET','POST'])
# @app.route('/index', methods=['GET','POST'])
def index():
    form = AminoacidSequencesForm()
    if request.method == "POST" and form.validate_on_submit():
        predictions, protInfo,_,_,_,_,_,_,_,_,_=predict(webApp=True, testSecFile=form.sequences.data,
          testNegAlFile="", #no need align negative test (used to evaluate the model)
          method="rbm",
          params={
              'rbm__n_iter': 20, 'rbm__n_components': 50, 'rbm__learning_rate': 0.1
            , "mod": "dt"
            , "mod_par": {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 1}
          },
          featToExtract=[True, False, False, False, False, False, False, True])

        predictions = ["Probable allergen" if p==1 else "Non-allergen" for p in predictions]

        return render_template('result.html',
                               allPred=predictions,
                               protInfo=protInfo)
    else:
        return render_template('index.html',
                           form=form)

@app.route('/train-allergens')
def allergen_data():
    dataset=getDataSet(True, True)
    return render_template('dataset.html',data=dataset, isAllergen=True, isTrain=True)

@app.route('/test-allergens')
def allergen_test_data():
    dataset=getDataSet(True, False)
    return render_template('dataset.html',data=dataset, isAllergen=True, isTrain=False)

@app.route('/train-non-allergens')
def nonallergen_data():
    dataset=getDataSet(False, True)
    return render_template('dataset.html', data=dataset, isAllergen=False, isTrain=True)

@app.route('/test-non-allergens')
def nonallergen_test_data():
    dataset=getDataSet(False, False)
    return render_template('dataset.html', data=dataset, isAllergen=False, isTrain=False)

def getDataSet(allergen=True, train=True):
    dataset=""
    filename="app/alignments/"

    if allergen and train:
        filename+="allergens_data_set.fasta"
    elif not allergen and train:
        filename+="non_allergens_data_set.fasta"
    if allergen and not train:
        filename+="test_allergens_data_set.fasta"
    else:
        filename+="test_nonallergens_data_set.fasta"
    
    lines=open(filename, "r").readlines()
    print(len(lines))
    for l in lines:
        dataset += Markup.escape(l) + Markup('<br />');
    return dataset

    
    


