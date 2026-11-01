"""
Treina o classificador de sentimento (modelo clássico) e exporta para JSON.

Dados: Brazilian E-Commerce Public Dataset by Olist (CC BY-NC-SA 4.0),
arquivo olist_order_reviews_dataset.csv. Rótulo derivado da nota:
  1-2 = negativo, 3 = neutro, 4-5 = positivo

Saídas:
  ml/metricas.json          métricas no conjunto de teste
  api/src/modelo/modelo.json pesos para a API em Node.js rodar sem Python
  data/teste.csv            avaliações de teste (usadas na comparação com o LLM)
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
CLASSES = ["negativo", "neutro", "positivo"]
SEMENTE = 42


def normalizar(texto: str) -> str:
    """Mesma normalização usada na API (minúsculas, sem acentos, espaços simples)."""
    t = unicodedata.normalize("NFD", str(texto).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", t).strip()


def rotulo(nota: int) -> str:
    return "negativo" if nota <= 2 else "neutro" if nota == 3 else "positivo"


def main() -> None:
    df = pd.read_csv(RAIZ / "data" / "olist_order_reviews_dataset.csv")
    df = df.dropna(subset=["review_comment_message"]).copy()
    df["texto"] = df["review_comment_message"].astype(str).str.strip()
    df = df[df["texto"].str.len() >= 3].drop_duplicates(subset=["texto"])
    df["sentimento"] = df["review_score"].astype(int).map(rotulo)

    treino, teste = train_test_split(df, test_size=0.2, random_state=SEMENTE, stratify=df["sentimento"])

    vetorizador = TfidfVectorizer(
        preprocessor=normalizar,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=3,
        max_features=12000,
        sublinear_tf=True,
    )
    x_treino = vetorizador.fit_transform(treino["texto"])
    x_teste = vetorizador.transform(teste["texto"])

    modelo = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    modelo.fit(x_treino, treino["sentimento"])
    previsto = modelo.predict(x_teste)

    metricas = {
        "dataset": "Olist Brazilian E-Commerce (avaliações com texto)",
        "amostras_treino": int(len(treino)),
        "amostras_teste": int(len(teste)),
        "acuracia": round(float(accuracy_score(teste["sentimento"], previsto)), 4),
        "f1_macro": round(float(f1_score(teste["sentimento"], previsto, average="macro")), 4),
        "por_classe": {
            k: {m: round(float(v), 4) for m, v in val.items()}
            for k, val in classification_report(teste["sentimento"], previsto, output_dict=True).items()
            if k in CLASSES
        },
        "matriz_confusao": {
            "ordem": CLASSES,
            "valores": confusion_matrix(teste["sentimento"], previsto, labels=CLASSES).tolist(),
        },
    }
    texto_metricas = json.dumps(metricas, ensure_ascii=False, indent=2)
    (RAIZ / "ml" / "metricas.json").write_text(texto_metricas, encoding="utf-8")
    (RAIZ / "api" / "src" / "modelo").mkdir(parents=True, exist_ok=True)
    (RAIZ / "api" / "src" / "modelo" / "metricas.json").write_text(texto_metricas, encoding="utf-8")

    # Exporta vocabulário, IDF e pesos para a API em TypeScript
    vocab = vetorizador.vocabulary_
    termos = sorted(vocab, key=vocab.get)
    classes = list(modelo.classes_)
    exportado = {
        "versao": 1,
        "classes": classes,
        "termos": termos,
        "idf": [round(float(v), 6) for v in vetorizador.idf_],
        "coef": [[round(float(v), 6) for v in linha] for linha in modelo.coef_],
        "intercepto": [round(float(v), 6) for v in modelo.intercept_],
        "ngramas": [1, 2],
    }
    destino = RAIZ / "api" / "src" / "modelo"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "modelo.json").write_text(json.dumps(exportado, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Probabilidades de referência para conferir se a API calcula igual
    amostra = teste.sample(200, random_state=SEMENTE)
    prob = modelo.predict_proba(vetorizador.transform(amostra["texto"]))
    referencia = [
        {"texto": t, "probabilidades": dict(zip(classes, [round(float(p), 6) for p in linha]))}
        for t, linha in zip(amostra["texto"], prob)
    ]
    (RAIZ / "api" / "tests" ).mkdir(parents=True, exist_ok=True)
    (RAIZ / "api" / "tests" / "referencia-python.json").write_text(json.dumps(referencia, ensure_ascii=False), encoding="utf-8")

    teste[["texto", "review_score", "sentimento"]].to_csv(RAIZ / "data" / "teste.csv", index=False)
    print(json.dumps({k: metricas[k] for k in ("amostras_treino", "amostras_teste", "acuracia", "f1_macro")}, indent=2))
    print(classification_report(teste["sentimento"], previsto, digits=3))


if __name__ == "__main__":
    np.random.seed(SEMENTE)
    main()
