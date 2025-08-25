from django.db import models
from django.utils import timezone

class LotteryDraw(models.Model):
    draw_date = models.DateField()
    draw_number = models.IntegerField(null=True, blank=True)
    number_1 = models.IntegerField()
    number_2 = models.IntegerField()
    number_3 = models.IntegerField()
    number_4 = models.IntegerField()
    number_5 = models.IntegerField()
    powerball = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-draw_date']
        unique_together = ['draw_date', 'draw_number']
    
    def __str__(self):
        return f"Draw {self.draw_number} on {self.draw_date}"
    
    def get_main_numbers(self):
        return [self.number_1, self.number_2, self.number_3, self.number_4, self.number_5]
    
    def get_all_numbers(self):
        return [self.number_1, self.number_2, self.number_3, self.number_4, self.number_5, self.powerball]

class Prediction(models.Model):
    MODEL_CHOICES = [
        ('frequency', 'Frequency Model'),
        ('pattern', 'Pattern Model'),
        ('clustering', 'Clustering Model'),
        ('regression', 'Regression Model'),
        ('neural', 'Neural Network Model'),
        ('time_series', 'Time Series Model'),
        ('ensemble', 'Ensemble Model'),
        ('bayesian', 'Bayesian Model'),
    ]
    
    prediction_date = models.DateField(default=timezone.now)
    draw_date = models.DateField()
    model_type = models.CharField(max_length=20, choices=MODEL_CHOICES)
    number_1 = models.IntegerField()
    number_2 = models.IntegerField()
    number_3 = models.IntegerField()
    number_4 = models.IntegerField()
    number_5 = models.IntegerField()
    powerball = models.IntegerField()
    confidence_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.model_type} prediction for {self.draw_date}"
    
    def get_main_numbers(self):
        return [self.number_1, self.number_2, self.number_3, self.number_4, self.number_5]
    
    def get_all_numbers(self):
        return [self.number_1, self.number_2, self.number_3, self.number_4, self.number_5, self.powerball]

class PredictionAccuracy(models.Model):
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE)
    actual_draw = models.ForeignKey(LotteryDraw, on_delete=models.CASCADE, null=True, blank=True)
    main_numbers_matched = models.IntegerField(default=0)
    powerball_matched = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Accuracy for {self.prediction}"
