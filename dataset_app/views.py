import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, arff
from django.shortcuts import render
from sklearn.model_selection import train_test_split

def index(request):
    context = {}

    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']

        # Cargar archivo ARFF correctamente
        data = arff.load(io.TextIOWrapper(uploaded_file.file, encoding='utf-8'))

        # Convertir a DataFrame
        df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
        context['dataframe_html'] = df.head(10).to_html(classes="table table-striped table-dark", index=False)

        # Separar el dataset (60/20/20)
        def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
            strat = df[stratify] if stratify else None
            train_set, test_set = train_test_split(df, test_size=0.4, random_state=rstate,
                                                   shuffle=shuffle, stratify=strat)
            strat = test_set[stratify] if stratify else None
            val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=rstate,
                                                 shuffle=shuffle, stratify=strat)
            return train_set, val_set, test_set

        # Intentar usar una columna categórica como estratificador
        strat_col = None
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 15:
                strat_col = col
                break

        try:
            train_set, val_set, test_set = train_val_test_split(df, stratify=strat_col)
        except Exception:
            train_set, val_set, test_set = train_val_test_split(df)

        # Buscar columnas categóricas o discretas
        categorical_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 15]

        figs = []
        if categorical_cols:
            # Usar solo la primera columna categórica
            col = categorical_cols[0]
            for dataset, title in [(df, 'Dataset completo'),
                                   (train_set, 'Train Set'),
                                   (val_set, 'Validation Set'),
                                   (test_set, 'Test Set')]:
                plt.figure(figsize=(7, 5))
                dataset[col].value_counts().plot(kind='bar', color='#4A90E2', edgecolor='black')
                plt.title(f'{title} — {col}', fontsize=13, fontweight='bold')
                plt.xlabel('Categorías')
                plt.ylabel('Frecuencia')
                plt.xticks(rotation=45)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                figs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        else:
            context['warning'] = "⚠️ No se encontraron columnas categóricas para graficar."

        context['graphs'] = figs

    return render(request, 'dataset_app/index.html', context)