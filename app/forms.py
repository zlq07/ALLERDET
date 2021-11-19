from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
from wtforms.widgets import TextArea

class AminoacidSequencesForm(FlaskForm):
    sequences = StringField('sequences', validators=[DataRequired()], widget=TextArea())    
