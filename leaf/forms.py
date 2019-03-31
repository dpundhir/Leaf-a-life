from django import forms
from .models import Leaf
class LeafForm(forms.ModelForm):
    class Meta:
        model = Leaf
        fields = ["leaf_image"]