// Modelo clássico: TF-IDF + regressão logística treinados em Python e
// exportados para JSON. Roda em Node.js puro, sem chamadas externas.
import modeloJson from './modelo/modelo.json' with { type: 'json' };
import { ngramas, tokenizar } from './texto.js';

export type Sentimento = 'negativo' | 'neutro' | 'positivo';

interface ModeloExportado {
  classes: Sentimento[];
  termos: string[];
  idf: number[];
  coef: number[][];
  intercepto: number[];
  ngramas: [number, number];
}

const modelo = modeloJson as unknown as ModeloExportado;
const indice = new Map(modelo.termos.map((t, i) => [t, i]));

export interface ResultadoClassico {
  sentimento: Sentimento;
  confianca: number;
  probabilidades: Record<Sentimento, number>;
}

export function classificar(texto: string): ResultadoClassico {
  // 1. contagem dos termos conhecidos
  const contagem = new Map<number, number>();
  for (const termo of ngramas(tokenizar(texto), ...modelo.ngramas)) {
    const i = indice.get(termo);
    if (i !== undefined) contagem.set(i, (contagem.get(i) ?? 0) + 1);
  }

  // 2. TF sublinear * IDF, depois normalização L2 (igual ao TfidfVectorizer)
  const pesos = new Map<number, number>();
  let norma = 0;
  for (const [i, tf] of contagem) {
    const w = (1 + Math.log(tf)) * modelo.idf[i];
    pesos.set(i, w);
    norma += w * w;
  }
  norma = Math.sqrt(norma) || 1;

  // 3. regressão logística multinomial: softmax(coef · x + intercepto)
  const logits = modelo.classes.map((_, c) => {
    let soma = modelo.intercepto[c];
    for (const [i, w] of pesos) soma += modelo.coef[c][i] * (w / norma);
    return soma;
  });
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const total = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / total);

  const melhor = probs.indexOf(Math.max(...probs));
  return {
    sentimento: modelo.classes[melhor],
    confianca: Number(probs[melhor].toFixed(4)),
    probabilidades: Object.fromEntries(modelo.classes.map((c, k) => [c, Number(probs[k].toFixed(4))])) as Record<Sentimento, number>,
  };
}
