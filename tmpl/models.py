from django.db import models

class User(models.Model):
    username = models.CharField(max_length = 30)
    img = models.FileField(upload_to = './upload/')

    def __unicode__(self):
        return self.username

# Create your models here.
