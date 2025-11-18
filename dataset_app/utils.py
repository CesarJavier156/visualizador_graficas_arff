import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib.figure import Figure
from io import BytesIO
import base64

def read_arff_to_dataframe(file_obj):
    """Convierte el archivo .arff en un DataFrame de pandas."""
    raw_bytes = file_obj.read()
    if isinstance(raw_bytes, bytes):
        raw = raw_bytes.decode('utf-8', errors='ignore')
    else:
        raw = raw_bytes

    dataset = arff.loads(raw)
    columns = [a[0] for a in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=columns)
    return df


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """Divide el dataset en 60% train, 20% val, 20% test."""
    strat = df[stratify] if stratify and stratify in df.columns else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )

    strat = test_set[stratify] if stratify and stratify in test_set.columns else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )

    return train_set, val_set, test_set


def plot_bar_distribution(df, column, title):
    """Crea una gráfica de barras mostrando la distribución de una columna categórica."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()

    if column not in df.columns:
        ax.text(0.5, 0.5, f"Columna '{column}' no encontrada", ha='center', va='center', fontsize=12)
    else:
        df[column].value_counts().plot(kind='bar', ax=ax, color='#3498db', alpha=0.7)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    return img_b64
