from django import forms
from .models import LotteryDraw, Prediction

class ImportDataForm(forms.Form):
    file = forms.FileField(
        label='Select Excel File',
        help_text='Upload an Excel file with Powerball draw data.',
        widget=forms.FileInput(attrs={'accept': '.xls,.xlsx'})
    )

class LotteryDrawForm(forms.ModelForm):
    class Meta:
        model = LotteryDraw
        fields = ['draw_date', 'draw_number', 'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'powerball']
        widgets = {
            'draw_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'draw_number': forms.NumberInput(attrs={'class': 'form-control'}),
            'number_1': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 50}),
            'number_2': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 50}),
            'number_3': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 50}),
            'number_4': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 50}),
            'number_5': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 50}),
            'powerball': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 20}),
        }

class DataImportForm(forms.Form):
    file = forms.FileField(
        label='Select a CSV or Excel file',
        help_text='File must have columns for draw date, numbers, and powerball',
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )

class PredictionConfigForm(forms.Form):
    MODEL_CHOICES = [
        ('frequency', 'Frequency Model'),
        ('pattern', 'Pattern Model'),
        ('clustering', 'Clustering Model'),
        ('regression', 'Regression Model'),
        ('neural', 'Neural Network Model'),
        ('time_series', 'Time Series Model'),
        ('ensemble', 'Ensemble Model (Combines All)'),
        ('bayesian', 'Bayesian Model'),
    ]
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    draw_date = forms.DateField(
        label='Prediction for Draw Date',
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'})
    )
    
    recency_weight = forms.FloatField(
        label='Recency Weight (0-1)',
        initial=0.7,
        min_value=0,
        max_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    lookback_periods = forms.IntegerField(
        label='Lookback Periods',
        initial=50,
        min_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control'}))