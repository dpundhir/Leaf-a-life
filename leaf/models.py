from django.db import models
from ckeditor.fields import RichTextField
from . import mod
# Create your models here.

class Leaf(models.Model):
    leaf_data = models.CharField(max_length = 2500,verbose_name = "Leaf Data")
    leaf_image = models.FileField(blank = True,null = True,verbose_name="Add Photo of the Leaf")
    
    # class Meta:
    #     ordering = ['-created_date']

# class Type(models.Model) :
#     leaf_type = 