from django import forms

class ARFFUploadForm(forms.Form):
    arff_file = forms.FileField(label='Archivo .arff')
