from django.shortcuts import render, redirect, get_object_or_404
from .forms import DatasetForm
from .models import Dataset
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

def select_models(request):
    datasets = Dataset.objects.all()
    return render(request, 'ml_tool/select_models.html', {'datasets': datasets})

def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('select_models')
    else:
        form = DatasetForm()
    return render(request, 'ml_tool/upload_dataset.html', {'form': form})

def perform_voting(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset')
        selected_models = request.POST.getlist('models')
        voting_type = request.POST.get('voting_type')

        if not selected_models:
            return render(request, 'ml_tool/results.html', {'error': 'No models selected. Please select at least one model.'})

        try:
            dataset = Dataset.objects.get(id=dataset_id)
            df = pd.read_csv(dataset.file.path)
        except Dataset.DoesNotExist:
            return render(request, 'ml_tool/results.html', {'error': 'Dataset not found!'})

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_map = {
            "logistic_regression": ('Logistic Regression', LogisticRegression(
                max_iter=int(request.POST.get('lr_max_iter', 100)),
                C=float(request.POST.get('lr_c', 1.0))
            )),
            "random_forest": ('Random Forest', RandomForestClassifier(
                n_estimators=int(request.POST.get('rf_n_estimators', 100)),
                max_depth=int(request.POST.get('rf_max_depth', 5)),
                random_state=42
            )),
            "svm": ('SVM', SVC(
                C=float(request.POST.get('svm_c', 1.0)),
                kernel=request.POST.get('svm_kernel', 'rbf'),
                probability=True
            )),
            "k_nearest_neighbors": ('KNN', KNeighborsClassifier(
                n_neighbors=int(request.POST.get('knn_n_neighbors', 5))
            )),
            "decision_tree": ('Decision Tree', DecisionTreeClassifier(
                max_depth=int(request.POST.get('dt_max_depth', 3)),
                random_state=42
            )),
        }
        models = [model_map[m] for m in selected_models if m in model_map]

        model_accuracies = {}
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_accuracies[name] = accuracy_score(y_test, y_pred) * 100
            except Exception as e:
                print(f"Error fitting {name}: {e}")
                model_accuracies[name] = f"Error: {e}"

        if voting_type in ['hard', 'soft']:
            try:
                voting_clf = VotingClassifier(estimators=models, voting=voting_type)
                voting_clf.fit(X_train, y_train)
                y_pred = voting_clf.predict(X_test)
                voting_accuracy = accuracy_score(y_test, y_pred) * 100
            except Exception as e:
                print(f"Error creating Voting Classifier: {e}")
                voting_accuracy = f"Error: {e}"
        else:
            voting_accuracy = None

        print("Model Accuracies:", model_accuracies)
        print("Voting Accuracy:", voting_accuracy)

        return render(request, 'ml_tool/results.html', {
            'model_accuracies': model_accuracies,
            'voting_accuracy': voting_accuracy,
        })

    return redirect('select_models')

def delete_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    dataset.delete()
    return redirect('select_models')
