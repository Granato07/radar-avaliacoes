# Radar de Avaliações

Análise de avaliações de clientes de e-commerce com **duas abordagens lado a lado**: um modelo clássico de Machine Learning treinado por mim e um LLM (Google Gemini) com saída estruturada. A API compara as duas no mesmo texto, e o dashboard (feito com **v0**) mostra o resultado.

O objetivo é responder uma pergunta prática: **quando vale pagar por um LLM e quando um modelo pequeno resolve?**

## O que ele faz

Você manda o texto de uma avaliação e recebe:

| | Modelo clássico | LLM (Gemini) |
|---|---|---|
| Sentimento (negativo / neutro / positivo) | ✅ com probabilidade | ✅ |
| Categoria do problema (atraso, produto errado, danificado…) | — | ✅ |
| Urgência (baixa / média / alta) | — | ✅ |
| Resumo em uma frase | — | ✅ |
| Sugestão de resposta ao cliente | — | ✅ |
| Custo por requisição | zero | cota da API |
| Tempo de resposta | < 1 ms | centenas de ms |

## Arquitetura

```
ml/treinar.py ──► modelo.json (vocabulário, IDF, pesos)
                       │
Dashboard (v0 / Next.js) ──► API Express + TypeScript
                                ├─ /api/analisar ─► classificador em TS puro (mesmos números do scikit-learn)
                                │                └► Gemini (JSON validado com zod)
                                ├─ /api/metricas
                                └─ /api/saude
```

- **Treino em Python, inferência em TypeScript.** O `treinar.py` exporta vocabulário, IDF e coeficientes para JSON; o `classico.ts` refaz o TF-IDF e a regressão logística em TS. Um teste compara as probabilidades com as do scikit-learn em 200 avaliações reais (diferença máxima < 0,001), então não precisa de servidor Python em produção.
- **Resiliência.** Se o modelo do Gemini estiver lento (limite de 12 s) ou fora do ar, a API tenta o próximo modelo da lista; a resposta informa qual modelo respondeu.
- **LLM com contrato.** O Gemini responde em JSON com `responseSchema`, e a API ainda valida tudo com zod. Se vier fora do formato, a API devolve erro em vez de repassar lixo.
- **Proteção contra prompt injection.** O texto do cliente vai entre `<avaliacao>` e a instrução de sistema manda tratar esse conteúdo só como dado.
- **Segurança.** A chave do Gemini fica só no servidor (variável de ambiente), CORS com lista de origens permitidas, limite de requisições por IP, corpo limitado a 8 KB e texto validado (3 a 1000 caracteres).

## Dados e modelo clássico

- **Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0), arquivo `olist_order_reviews_dataset.csv`. Usei as 36 mil avaliações com texto e sem duplicatas.
- **Rótulos:** nota 1–2 = negativo, 3 = neutro, 4–5 = positivo.
- **Modelo:** TF-IDF (unigramas e bigramas, 12 mil termos, sem acentos, TF sublinear) + regressão logística com classes balanceadas.
- **Divisão:** 80/20 estratificada (29.062 treino / 7.266 teste).

### Resultados no conjunto de teste

| Métrica | Valor |
|---|---|
| Acurácia | **78,9%** |
| F1 macro | 0,657 |
| F1 negativo | 0,781 |
| F1 neutro | 0,298 |
| F1 positivo | 0,893 |

Matriz de confusão (linhas = real, colunas = previsto):

| | negativo | neutro | positivo |
|---|---|---|---|
| **negativo** | 1724 | 364 | 89 |
| **neutro** | 273 | 256 | 139 |
| **positivo** | 240 | 431 | 3750 |

**O ponto fraco é o neutro.** Nota 3 mistura elogio com reclamação ("chegou rápido, mas a qualidade é fraca"), e um modelo de palavras não entende esse "mas". É exatamente aí que o LLM deve ajudar, e o script de comparação mede isso.

## Clássico x LLM

`npm run comparar -- 60` roda a mesma amostra estratificada do teste (20 de cada classe) nos dois modos e salva `ml/comparacao.json` com acurácia, recall do neutro, tempo médio e as categorias encontradas pelo LLM.

<!-- resultados da comparação entram aqui -->

## Como rodar

### API

```bash
cd api
npm install
cp .env.example .env   # coloque sua GEMINI_API_KEY (Google AI Studio)
npm run dev            # http://localhost:3333
npm test               # 13 testes (paridade com Python, validação, LLM simulado, modelo reserva, timeout, rate limit)
```

```bash
curl -X POST http://localhost:3333/api/analisar \
  -H "Content-Type: application/json" \
  -d '{"texto": "Comprei há 20 dias e até agora nada. Ninguém responde.", "modo": "ambos"}'
```

`modo` aceita `classico`, `llm` ou `ambos` (padrão). Sem chave do Gemini o modo clássico continua funcionando.

### Retreinar o modelo

```bash
pip install -r ml/requirements.txt
# baixe olist_order_reviews_dataset.csv do Kaggle para data/
python ml/treinar.py
```

O script atualiza `api/src/modelo/`, as métricas e a referência usada no teste de paridade.

## Deploy

- **API:** Vercel, com *Root Directory* `api` (Express detectado automaticamente pelo `src/app.ts`). Variáveis: `GEMINI_API_KEY`, `ORIGENS_PERMITIDAS` (URL do dashboard) e, se quiser, `GEMINI_MODEL` (lista de modelos separada por vírgula, tentados em ordem), `GEMINI_TIMEOUT_MS` e `LIMITE_POR_MINUTO`.
- **Dashboard:** gerado no v0 e publicado na Vercel. Ele só conversa com a API; nenhuma chave fica no navegador.

## Stack

Python · scikit-learn · pandas · TypeScript · Node.js · Express 5 · zod · Vitest · Supertest · Google Gemini API · v0 · Next.js · Vercel

## Autor

Daniel Granato · [Portfólio](https://danielgranato.vercel.app) · [GitHub](https://github.com/Granato07)
