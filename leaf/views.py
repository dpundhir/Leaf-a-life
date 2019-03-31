import pandas as pd
import numpy as np 
import operator
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle
from hack_bpit.apply import *




from django.shortcuts import render,HttpResponse,redirect,get_object_or_404,reverse
from .forms import LeafForm
from django.contrib import messages
from .models import Leaf
# from apply import .



x=0
# Create your views here.

def index(request):
    leafs = Leaf.objects.all()
    return render(request,"index.html",{"leafs":leafs})

def about(request):
    return render(request,"about.html")

def contact(request):
    return render(request,"contact.html")

def newLeaf(request):
    form = LeafForm(request.POST or None,request.FILES or None)

    if form.is_valid():
        leaf = form.save(commit=False)
        
        leaf.save()

        messages.success(request,"Leaf added successfully")
        return redirect(process)
    return render(request,"addleaf.html",{"form":form})

# def newLeaf(request):
#     form = LeafForm(request.POST or None,request.FILES or None)
#     leaf = form.save(commit=False)
#     leaf.save()

#     return render(request,"addleaf.html",{"form":form})


# def process(request) :
#     details = Leaf.objects.all()[Leaf.objects.count()-1]
#     data=""
#     # for detail in details :
#     data = details.leaf_data
#     lst = data.split(',')
#     results = [float(i) for i in lst]
#     with open('leaf/trained_model','rb') as f:
#         model = pickle.load(f)
#     a = model[0].predict_proba(np.reshape(results,(-1,192)))

#     max_index,_ = max(enumerate(a[0]), key=operator.itemgetter(1))

#     details.leaf_data = str(max_index)
    details1= {
        "details" : details,
    }
    # print("Category is :- ",max_index)
    return render(request,"details.html",details1)


def process(request):
    details = Leaf.objects.all()[Leaf.objects.count()-1]
    i1 = details.leaf_image.path

    # from PIL import Image
    # img = Image.open(details.leaf_image)

    disease = analysis(i1)    
    details.leaf_data = disease
    details1= {
        "details" : details,
    }
    return render(request,"details.html",details1)