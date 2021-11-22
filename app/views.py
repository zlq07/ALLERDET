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
        predictions, protInfo=predict(webApp=True, testSecFile=form.sequences.data)

        return render_template('result.html',
                               allPred=predictions,
                               protInfo=protInfo)
    else:
        return render_template('index.html',
                           form=form)

@app.route('/alldata')
def allergen_data():
    dataset=getDataSet()
    return render_template('dataset.html',data=dataset, isAllergen=True)

@app.route('/nalldata')
def nonallergen_data():
    dataset=getDataSet(False)
    return render_template('dataset.html', data=dataset, isAllergen=False)

def getDataSet(allergen=True):
    dataset=""
    filename="app/alignments/"
    filename+="allergens_data_set.fasta" if allergen else "non_allergens_data_set.fasta"
    
    lines=open(filename, "r").readlines()
    print(len(lines))
    for l in lines:
        dataset += Markup.escape(l) + Markup('<br />');
    return dataset

    
    


