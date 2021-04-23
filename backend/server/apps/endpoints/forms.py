from django import forms

class PredictForm(forms.Form):
    text = forms.CharField(label='Text | Текст', max_length=250)
