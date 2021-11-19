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

import os
import sys

#sys.path.insert(0, os.path.dirname(__file__))

from app import app as application

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    application.run(debug=False, host="0.0.0.0", port=port) #host visible desde internet