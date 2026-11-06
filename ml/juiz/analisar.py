"""Compara estrela, modelo clássico e dois LLMs contra os rótulos do juiz.

Entradas (nesta pasta): amostra.csv, rotulos_juiz.json, respostas.json (gerado por api/scripts/coletar.ts)
Saída: resultado.json e um resumo no terminal.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score

PASTA = Path(__file__).parent
CLASSES = ["negativo", "neutro", "positivo"]
MODELOS = {"openai/gpt-oss-20b": "llm_20b", "openai/gpt-oss-120b": "llm_120b"}
PRECOS = json.loads((PASTA / "precos.json").read_text())  # US$ por 1M tokens
ESCALAS = [1_000, 100_000, 1_000_000]

amostra = pd.read_csv(PASTA / "amostra.csv")
juiz = json.loads((PASTA / "rotulos_juiz.json").read_text())["rotulos"]
respostas = json.loads((PASTA / "respostas.json").read_text())

amostra["juiz"] = amostra.id.map(lambda i: juiz[str(i)]["juiz"])
amostra["conf_juiz"] = amostra.id.map(lambda i: juiz[str(i)]["confianca"])
amostra = amostra.rename(columns={"sentimento": "estrela"})

linhas = {}
for r in respostas:
    d = linhas.setdefault(r["id"], {})
    if "erro" in r:
        continue
    col = MODELOS[r["modelo"]]
    d[col] = r["llm"]["sentimento"]
    d[f"{col}_tok_in"] = r["llm"]["uso"]["tokens_entrada"]
    d[f"{col}_tok_out"] = r["llm"]["uso"]["tokens_saida"]
    d[f"{col}_ms"] = r["llm"]["tempo_ms"]
    d["classico"] = r["classico"]["sentimento"]
    d["p_positivo"] = r["classico"]["probabilidades"]["positivo"]
    d["conf_classico"] = r["classico"]["confianca"]
df = amostra.merge(pd.DataFrame.from_dict(linhas, orient="index"), left_on="id", right_index=True, how="left")
completos = df.dropna(subset=["classico", "llm_20b", "llm_120b"])


def metricas(y_true, y_pred):
    return {
        "acuracia": round(accuracy_score(y_true, y_pred), 3),
        "f1_macro": round(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0), 3),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 3),
        "acerto_negativos": round(recall_score(y_true, y_pred, labels=["negativo"], average="macro", zero_division=0), 3),
        "acerto_neutros": round(recall_score(y_true, y_pred, labels=["neutro"], average="macro", zero_division=0), 3),
        "acerto_positivos": round(recall_score(y_true, y_pred, labels=["positivo"], average="macro", zero_division=0), 3),
    }


fontes = {"estrela": "estrela", "classico": "classico", "llm_20b": "llm_20b", "llm_120b": "llm_120b"}
contra_juiz = {nome: metricas(completos.juiz, completos[col]) for nome, col in fontes.items()}
alta = completos[completos.conf_juiz == "alta"]
contra_juiz_alta = {nome: metricas(alta.juiz, alta[col]) for nome, col in fontes.items()}
# Referência: o que a comparação antiga media (contra a estrela)
contra_estrela = {nome: metricas(completos.estrela, completos[col]) for nome, col in fontes.items() if nome != "estrela"}

# Custo e tempo medidos
custo = {}
for modelo, col in MODELOS.items():
    tin, tout = completos[f"{col}_tok_in"].mean(), completos[f"{col}_tok_out"].mean()
    p = PRECOS[modelo]
    por_chamada = (tin * p["entrada"] + tout * p["saida"]) / 1e6
    custo[col] = {
        "tokens_entrada_medio": round(tin, 1),
        "tokens_saida_medio": round(tout, 1),
        "custo_por_chamada_usd": round(por_chamada, 8),
        "custo_usd_por_escala": {str(n): round(por_chamada * n, 2) for n in ESCALAS},
        "tempo_ms_mediana": int(completos[f"{col}_ms"].median()),
        "tempo_ms_p90": int(completos[f"{col}_ms"].quantile(0.9)),
    }

# Quanto custaria usar um Claude como juiz via API (estimativa: mesmo tamanho de entrada medido no Groq
# + a rubrica, ~250 tokens, e ~10 tokens de saída, já que o juiz devolve só o rótulo)
tin_juiz = custo["llm_20b"]["tokens_entrada_medio"] + 250
juiz_custo = {}
for m in ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]:
    p = PRECOS[m]
    c = (tin_juiz * p["entrada"] + 10 * p["saida"]) / 1e6
    juiz_custo[m] = {"por_avaliacao_usd": round(c, 6), "amostra_150_usd": round(c * 150, 3), "por_1_milhao_usd": round(c * 1e6, 0)}

# Triagem: clássico decide sozinho quando tem certeza de que é positiva; o resto vai para o LLM (20b)
triagem = []
custo_20b = custo["llm_20b"]["custo_por_chamada_usd"]
for limiar in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    pula = (completos.classico == "positivo") & (completos.conf_classico >= limiar)
    final = np.where(pula, "positivo", completos.llm_20b)
    reclamacoes = completos.juiz == "negativo"
    triagem.append({
        "limiar": limiar,
        "llm_evitado": round(pula.mean(), 3),
        "acuracia": round(accuracy_score(completos.juiz, final), 3),
        "acerto_negativos": round(recall_score(completos.juiz, final, labels=["negativo"], average="macro", zero_division=0), 3),
        "reclamacoes_perdidas": int((pula & reclamacoes).sum()),
        "custo_por_milhao_usd": round(custo_20b * (1 - pula.mean()) * 1e6, 2),
    })

# Prompt v2 (mesmo modelo 20b, prompt com a definição de sentimento do atendimento)
prompt_v2 = None
arq_v2 = PASTA / "respostas_v2.json"
if arq_v2.exists():
    v2 = {r["id"]: r["llm"] for r in json.loads(arq_v2.read_text()) if "erro" not in r}
    base = completos[completos.id.isin(v2)]
    pred_v2 = base.id.map(lambda i: v2[i]["sentimento"])
    tin = float(np.mean([v2[i]["uso"]["tokens_entrada"] for i in base.id]))
    tout = float(np.mean([v2[i]["uso"]["tokens_saida"] for i in base.id]))
    p = PRECOS["openai/gpt-oss-20b"]
    por_chamada = (tin * p["entrada"] + tout * p["saida"]) / 1e6
    prompt_v2 = {
        "n": len(base),
        "v1": metricas(base.juiz, base.llm_20b),
        "v2": metricas(base.juiz, pred_v2),
        "v2_contra_juiz_confianca_alta": metricas(base[base.conf_juiz == "alta"].juiz, pred_v2[base.conf_juiz == "alta"]),
        "tokens_entrada_medio": round(tin, 1),
        "tokens_saida_medio": round(tout, 1),
        "custo_por_milhao_usd": round(por_chamada * 1e6, 2),
        "mudou": int((pred_v2 != base.llm_20b).sum()),
    }
    # Triagem (limiar 0,8) combinada com o prompt v2: o padrão atual da API
    pula = (base.classico == "positivo") & (base.conf_classico >= 0.8)
    final = np.where(pula, "positivo", pred_v2)
    prompt_v2["triagem_0_8_com_v2"] = {
        "acuracia": round(accuracy_score(base.juiz, final), 3),
        "acerto_negativos": round(recall_score(base.juiz, final, labels=["negativo"], average="macro", zero_division=0), 3),
        "reclamacoes_perdidas": int((pula & (base.juiz == "negativo")).sum()),
        "custo_por_milhao_usd_amostra": round(por_chamada * (1 - pula.mean()) * 1e6, 2),
        "custo_por_milhao_usd_distribuicao_real": round(por_chamada * (1 - 0.416) * 1e6, 2),
    }

# Validação humana do juiz (40 avaliações rotuladas às cegas pelo autor)
validacao_humana = None
if (PASTA / "humano.csv").exists():
    h = pd.read_csv(PASTA / "humano.csv")
    validacao_humana = {
        "n": len(h),
        "concordancia_humano_juiz": round((h.humano == h.juiz).mean(), 3),
        "kappa_humano_juiz": round(cohen_kappa_score(h.humano, h.juiz), 3),
        "concordancia_humano_estrela": round((h.humano == h.estrela).mean(), 3),
        "limite_inferior_95": 0.912 if (h.humano == h.juiz).all() else None,
        "observacao": "40 de 40; só 4 dos 40 eram casos que o juiz marcou como ambíguos",
    }

resultado = {
    "amostra": f"{len(completos)} avaliações (50 por classe de estrela, random_state=7)",
    "juiz": "Claude Opus 5.5, leitura cega do texto (rubrica.md)",
    "validacao_humana": validacao_humana,
    "distribuicao_real_triagem": {"avaliacoes": 7266, "llm_evitado": 0.416, "limiar": 0.8, "nota_1_2_puladas": 12, "dessas_reclamacoes_reais": 0},
    "estrela_x_juiz": pd.crosstab(completos.estrela, completos.juiz).to_dict(),
    "contra_juiz": contra_juiz,
    "contra_juiz_so_confianca_alta": {"n": len(alta), **contra_juiz_alta},
    "contra_estrela": contra_estrela,
    "custo_e_tempo": custo,
    "custo_estimado_juiz_claude": juiz_custo,
    "triagem": triagem,
    "prompt_v2": prompt_v2,
    "precos_usd_por_milhao": PRECOS,
}
texto = json.dumps(resultado, ensure_ascii=False, indent=1)
(PASTA / "resultado.json").write_text(texto)
(PASTA.parent.parent / "api" / "src" / "modelo" / "avaliacao.json").write_text(texto)  # servido em /api/avaliacao

print(f"{len(completos)} avaliações completas")
print("\nContra o juiz:")
print(pd.DataFrame(contra_juiz).T.to_string())
print(f"\nContra o juiz, só confiança alta (n={len(alta)}):")
print(pd.DataFrame(contra_juiz_alta).T.to_string())
print("\nContra a estrela:")
print(pd.DataFrame(contra_estrela).T.to_string())
print("\nCusto e tempo:")
print(json.dumps(custo, indent=1))
print("\nJuiz Claude via API (estimativa):", json.dumps(juiz_custo, indent=1))
print("\nPrompt v2:", json.dumps(prompt_v2, indent=1))
print("\nTriagem:")
print(pd.DataFrame(triagem).to_string(index=False))
