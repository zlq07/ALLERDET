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
Fecha: 30-06-2016 (ultima modificacion: 27/04/2017)
----------------------------------------------------------
"""

from flask import Flask

app = Flask(__name__)
app.config.from_object('config')

from app import views

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000) #host visible desde internet