from django.contrib import admin
from .models import Leaf
# Register your models here.
@admin.register(Leaf)
class LeafAdmin(admin.ModelAdmin):

    list_display = ["leaf_data","leaf_image"]

    class Meta:
        model = Leaf