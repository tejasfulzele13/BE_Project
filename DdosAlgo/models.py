
from django.db import models

class AttackPrediction(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    source_port = models.IntegerField()
    destination_port = models.IntegerField()
    packet_size = models.FloatField()
    packet_rate = models.FloatField()
    connection_duration = models.FloatField()
    protocol = models.CharField(max_length=50)
    request_type = models.CharField(max_length=50)
    anomaly_score = models.FloatField()
    prediction = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.timestamp} - {self.prediction}"
