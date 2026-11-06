# Radar de Avaliações

Análise de avaliações de clientes de e-commerce com **duas abordagens lado a lado**: um modelo clássico de Machine Learning treinado por mim e um LLM (Groq, com Google Gemini como reserva) com saída estruturada. A API compara as duas no mesmo texto, e o dashboard (feito com **v0**) mostra o resultado.

O objetivo é responder uma pergunta prática: **quando vale pagar por um LLM e quando um modelo pequeno resolve?** Para isso o projeto mede acerto, tokens, custo e tempo de cada abordagem contra um gabarito confiável: um juiz (Claude Opus 5.5) validado por rotulagem humana. Os detalhes estão em [Como avaliei](#como-avaliei).

## O que ele faz

Você manda o texto de uma avaliação e recebe:

| | Modelo clássico | LLM (Groq / Gemini) |
|---|---|---|
| Sentimento (negativo / neutro / positivo) | ✅ com probabilidade | ✅ |
| Categoria do problema (atraso, produto errado, danificado…) | — | ✅ |
| Urgência (baixa / média / alta) | — | ✅ |
| Resumo em uma frase | — | ✅ |
| Sugestão de resposta ao cliente | — | ✅ |
| Tokens e custo por chamada | zero | ~695 tokens · US$ 0,00008 (gpt-oss-20b, prompt v2) |
| Tempo de resposta | ~0,02 ms | ~0,5 s (mediana) |

## Arquitetura

```
ml/treinar.py ──► modelo.json (vocabulário, IDF, pesos)
                       │
Dashboard (v0 / Next.js) ──► API Express + TypeScript
                                ├─ /api/analisar ─► classificador em TS puro (mesmos números do scikit-learn)
                                │                └► LLM: Groq → Gemini (JSON validado com zod)
                                ├─ /api/metricas   (métricas do modelo clássico)
                                ├─ /api/avaliacao  (resultado da avaliação com juiz, para o dashboard)
                                └─ /api/saude
```

- **Treino em Python, inferência em TypeScript.** O `treinar.py` exporta vocabulário, IDF e coeficientes para JSON; o `classico.ts` refaz o TF-IDF e a regressão logística em TS. Um teste compara as probabilidades com as do scikit-learn em 200 avaliações reais (diferença máxima < 0,001), então não precisa de servidor Python em produção.
- **Resiliência entre provedores.** A API tenta os modelos do Groq e, se estiverem lentos (limite de 10 s), fora do ar ou sem cota, cai para o Gemini. A resposta informa qual provedor e modelo responderam. Isso nasceu de um problema real: no plano gratuito do Gemini as requisições passaram a dar timeout e 503 por sobrecarga, então o LLM principal virou o Groq e o Gemini ficou de reserva.
- **Triagem para cortar custo.** No modo `triagem`, o modelo clássico responde sozinho quando tem certeza (≥ 0,8) de que a avaliação é positiva, porque elogio não precisa de categoria, urgência nem resposta. O resto vai para o LLM. Na distribuição real isso evita 42% das chamadas sem perder reclamações (ver [Triagem](#5-triagem-corta-42-das-chamadas-sem-perder-reclamação)).
- **Tokens e custo em toda resposta.** Cada análise do LLM devolve `uso: { tokens_entrada, tokens_saida, custo_usd }`, calculado com a tabela de preços em `src/precos.ts`.
- **Prompt versionado.** O prompt v2 venceu o v1 na avaliação com juiz e é o padrão; o v1 continua disponível para comparação.
- **LLM com contrato.** O Groq responde com *structured outputs* em modo strict (`json_schema`) e o Gemini com `responseSchema`; mesmo assim a API valida tudo com zod. Se vier fora do formato, a API devolve erro em vez de repassar lixo.
- **Proteção contra prompt injection.** O texto do cliente vai entre `<avaliacao>` e a instrução de sistema manda tratar esse conteúdo só como dado.
- **Segurança.** As chaves dos LLMs ficam só no servidor (variáveis de ambiente), CORS com lista de origens permitidas, limite de requisições por IP, corpo limitado a 8 KB e texto validado (3 a 1000 caracteres).

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

**O ponto fraco é o neutro.** Nota 3 mistura elogio com reclamação ("chegou rápido, mas a qualidade é fraca"). A comparação abaixo mostra que boa parte desse erro vem do próprio rótulo.

## Como avaliei

### 1. A nota em estrelas não serve como gabarito

A primeira comparação usava a estrela como resposta certa (1–2 = negativo, 3 = neutro, 4–5 = positivo). Lendo os erros, ficou claro que o problema estava no gabarito: muita avaliação de 3 estrelas é uma reclamação clara ("Recebi o produto errado", "Não recebi a mercadoria"), e algumas de 1 estrela são elogios de quem clicou errado ("Serviço muito bom").

Então troquei o gabarito: um modelo mais forte lê o **texto** e diz o sentimento.

### 2. Juiz: um LLM mais forte, às cegas, validado por um humano

- **Amostra:** 150 avaliações do conjunto de teste, 50 de cada classe de estrela (`random_state=7`), em ordem embaralhada. Arquivo: [`ml/juiz/amostra.csv`](ml/juiz/amostra.csv).
- **Juiz:** Claude Opus 5.5 (Anthropic) rotulou as 150 lendo **só o texto**, sem ver a nota nem a resposta de nenhum modelo, seguindo uma rubrica escrita antes ([`rubrica.md`](ml/juiz/rubrica.md)). Os rótulos foram feitos numa sessão de trabalho com o Claude, não por chamada de API. Resultado em [`rotulos_juiz.json`](ml/juiz/rotulos_juiz.json).
- **Validação humana:** eu rotulei 40 dessas avaliações (sorteadas) às cegas, com a mesma rubrica, sem ver os rótulos do juiz. Concordei com o juiz em **40 de 40** (kappa 1,0). Com 40 casos, a concordância real fica acima de 91% com 95% de confiança, e só 4 desses 40 eram casos ambíguos. Ou seja: o juiz segue bem a rubrica, mas isso não prova que ele seja infalível. Arquivo: [`humano.csv`](ml/juiz/humano.csv).
- **O juiz confirmou o problema do gabarito:** das 50 avaliações de 3 estrelas, **29 são reclamações** pelo texto. A estrela concorda com o juiz em só 70% dos casos.

### 3. Resultado contra o juiz

| | Acerto | Kappa | Reclamações encontradas | Neutros | Positivos |
|---|---|---|---|---|---|
| Estrela (gabarito antigo) | 70,0% | 0,55 | 59,5% | 66,7% | 89,4% |
| Modelo clássico | 74,0% | 0,60 | 67,1% | 66,7% | 89,4% |
| LLM pequeno (gpt-oss-20b) | **88,7%** | **0,81** | **96,2%** | 45,8% | 97,9% |
| LLM grande (gpt-oss-120b) | **89,3%** | **0,81** | **96,2%** | 45,8% | 100% |

*Kappa* é a concordância descontando a que aconteceria por acaso (0 = acaso, 1 = perfeito). Dados completos em [`ml/juiz/resultado.json`](ml/juiz/resultado.json).

**A forma de medir mudou a conclusão.** Contra a estrela, LLM e clássico pareciam empatados (67% contra 66%). Contra um gabarito confiável, o LLM acerta 89% e o clássico 74%, e o LLM encontra 96% das reclamações contra 67%. Se eu tivesse parado na estrela, teria concluído que o LLM não valia o custo.

**Modelo pequeno ou grande?** O 120b acertou praticamente o mesmo que o 20b (diferença de 1 avaliação, dentro da margem de erro) e custa 2,3 vezes mais. Para essa tarefa, o modelo pequeno é a escolha certa.

**Onde os LLMs ainda erram:** no neutro. Quase todos os erros são textos mistos ("produto bom, mas o frete é caro") classificados como negativos. Isso levou ao prompt v2 ([seção 6](#6-prompt-v2-mesma-ia-prompt-melhor-medido-contra-o-juiz)).

### 4. Custo, tokens e escala

Tokens medidos em cada chamada real (a API devolve `uso` na resposta). Preços de tabela pública consultados em 06/11/2026 ([`precos.json`](ml/juiz/precos.json)); confira a página oficial antes de orçar.

| | Tokens por chamada (entrada + saída) | Custo por chamada | 1 mil avaliações | 100 mil | 1 milhão | Tempo (mediana) |
|---|---|---|---|---|---|---|
| Modelo clássico | 0 | US$ 0 | US$ 0 | US$ 0 | US$ 0 | ~0,02 ms |
| gpt-oss-20b | 414 + 126 | US$ 0,000069 | US$ 0,07 | US$ 6,88 | US$ 68,76 | 478 ms |
| gpt-oss-120b | 414 + 157 | US$ 0,000156 | US$ 0,16 | US$ 15,64 | US$ 156,39 | 604 ms |
| Claude Opus 5.5 como juiz (estimativa) | ~664 + ~10 | US$ 0,0029 | US$ 2,86 | US$ 286 | US$ 2.855 | — |

- Os números do 20b e do 120b são com o prompt v1. Com o v2, que é o padrão hoje, o 20b gasta 571 + 125 tokens, ou US$ 80 por milhão (seção 6).
- **Por que o juiz não roda em produção:** ele custa ~40 vezes mais que o gpt-oss-20b por avaliação. A conta fecha usando o modelo caro só para **avaliar uma amostra pequena**: rotular as 150 avaliações via API custaria cerca de US$ 0,43.
- **Memória:** o modelo clássico é um JSON de 600 KB e ocupa uns 36 MB de memória no processo Node. Os LLMs rodam no provedor, então não ocupam memória na API, mas custam por token. Se fossem hospedados por conta própria, precisariam de GPU com dezenas de GB de memória (o 120b ainda mais que o 20b).
- **O que mais pesa no custo é a saída:** o token de saída custa 4 vezes mais que o de entrada, e o resumo mais a resposta sugerida são a maior parte da saída. Se o sistema só precisasse do sentimento, daria para pedir só ele e cortar a maior parte do custo.

### 5. Triagem: corta 42% das chamadas sem perder reclamação

Ideia: o modelo clássico é de graça e muito bom em reconhecer elogio óbvio. Quando ele tem certeza de que a avaliação é positiva, não precisa chamar o LLM.

| Limiar de confiança | LLM evitado (amostra) | Acerto | Reclamações perdidas | Custo por 1 milhão |
|---|---|---|---|---|
| 0,50 | 31% | 88,0% | 1 | US$ 47 |
| 0,70 | 29% | 88,0% | 1 | US$ 49 |
| **0,80 (escolhido)** | **25%** | **89,3%** | **0** | **US$ 51** |
| 0,90 | 21% | 89,3% | 0 | US$ 54 |
| sem triagem | 0% | 88,7% | 0 | US$ 69 |

- Escolhi **0,8**: é o menor limiar que não deixou passar nenhuma reclamação, e o acerto não caiu.
- A amostra é equilibrada (1/3 positivas). **Na distribuição real**, rodando o clássico nas 7.266 avaliações do teste, a triagem com 0,8 evita **42% das chamadas** ao LLM.
- Nessas 7.266, a triagem deixou passar 12 avaliações de 1–2 estrelas. Li as 12: todas são elogios com a estrela errada ("Serviço muito bom", "Um excelente site. Super recomendo."). Nenhuma reclamação real foi perdida.

### 6. Prompt v2: mesma IA, prompt melhor, medido contra o juiz

A análise de erros mostrou um padrão: o LLM chamava de **negativo** quase todo texto **misto** ("produto bom, mas o frete é caro"). O prompt v1 nem dizia o que era neutro. O v2 define o sentimento do jeito que o time de atendimento lê, com a mesma regra da rubrica: vale o que **predomina**, e texto que equilibra elogio e ressalva é neutro. Não usei nenhuma avaliação da amostra como exemplo no prompt. Os dois estão em [`api/src/llm.ts`](api/src/llm.ts) (`PROMPTS`).

| gpt-oss-20b | Acerto | Kappa | Reclamações encontradas | Neutros | Tokens de entrada | Custo por 1 milhão |
|---|---|---|---|---|---|---|
| Prompt v1 | 88,7% | 0,81 | 96,2% | 45,8% | 414 | US$ 68,76 |
| **Prompt v2** | **93,3%** | **0,89** | 96,2% | **75,0%** | 571 | US$ 80,15 |

- O v2 acertou 7 avaliações a mais e **não perdeu nenhuma reclamação**. O ganho veio todo do neutro.
- **Isso custa:** o prompt maior gasta 157 tokens a mais por chamada, cerca de **17% a mais**. Por 4,6 pontos de acerto, achei que valia, e o v2 virou o padrão da API (o v1 continua disponível com `"prompt": "v1"`).
- **Configuração final (triagem 0,8 + prompt v2):** acerto de 92,0%, nenhuma reclamação perdida e cerca de **US$ 47 por milhão de avaliações** na distribuição real, contra US$ 80 sem triagem.
- Para comparar: trocar o 20b pelo 120b custaria 2,3 vezes mais e não melhorou nada. **Melhorar o prompt rendeu mais do que pagar por um modelo maior.**

### 7. Limitações

- **Amostra pequena.** Com 150 avaliações, a margem de erro do acerto é de cerca de ±5 a 7 pontos percentuais. Diferenças menores que isso, como 20b contra 120b, não são conclusivas.
- **Um juiz só.** O juiz foi validado em 40 casos, quase todos fáceis. Os casos mistos são justamente onde humanos também discordam.
- **Prompt ajustado na mesma amostra.** O prompt v2 foi escrito a partir da rubrica, sem exemplos da amostra, mas foi medido nas mesmas 150 avaliações que revelaram o problema. O ganho real pode ser um pouco menor; o próximo passo seria confirmar numa amostra nova.
- **Preços mudam.** Os custos usam tabelas públicas de novembro de 2026.

A primeira comparação, de 60 avaliações contra a estrela, continua em [`ml/comparacao.json`](ml/comparacao.json) como registro.

## Como rodar

### API

```bash
cd api
npm install
cp .env.example .env   # coloque GROQ_API_KEY (console.groq.com) e/ou GEMINI_API_KEY
npm run dev            # http://localhost:3333
npm test               # 22 testes (paridade com Python, validação, LLM simulado, troca de provedor, timeout, rate limit, tokens/custo, triagem, prompt)
```

```bash
curl -X POST http://localhost:3333/api/analisar \
  -H "Content-Type: application/json" \
  -d '{"texto": "Comprei há 20 dias e até agora nada. Ninguém responde.", "modo": "ambos"}'
```

`modo` aceita `classico`, `llm`, `ambos` (padrão) ou `triagem`. Campos opcionais: `modelo` (um dos modelos configurados no servidor, usado sozinho, sem cair para outro) e `prompt` (`v1` ou `v2`, padrão `v2`). Sem chave de LLM o modo clássico continua funcionando.

### Reproduzir a avaliação

```bash
cd api && API_URL=https://radar-avaliacoes-api.vercel.app npm run coletar   # gera ml/juiz/respostas.json
cd .. && python ml/juiz/analisar.py                                          # gera ml/juiz/resultado.json e api/src/modelo/avaliacao.json
```

### Retreinar o modelo

```bash
pip install -r ml/requirements.txt
# baixe olist_order_reviews_dataset.csv do Kaggle para data/
python ml/treinar.py
```

O script atualiza `api/src/modelo/`, as métricas e a referência usada no teste de paridade.

## Deploy

- **API:** Vercel, com *Root Directory* `api` (Express detectado automaticamente pelo `src/app.ts`). Variáveis: `GROQ_API_KEY` e/ou `GEMINI_API_KEY`, `ORIGENS_PERMITIDAS` (URL do dashboard) e, se quiser, `GROQ_MODEL` / `GEMINI_MODEL` (listas separadas por vírgula, tentadas em ordem), `LLM_TIMEOUT_MS`, `LIMITE_POR_MINUTO` e `LIMIAR_TRIAGEM` (padrão 0,8).
- **Dashboard:** gerado no v0 e publicado na Vercel. Ele só conversa com a API; nenhuma chave fica no navegador.

## Stack

Python · scikit-learn · pandas · LLM-as-judge (Claude) · TypeScript · Node.js · Express 5 · zod · Vitest · Supertest · Groq API · Google Gemini API · v0 · Next.js · Vercel

## Autor

Daniel Granato · [Portfólio](https://danielgranato.vercel.app) · [GitHub](https://github.com/Granato07)
