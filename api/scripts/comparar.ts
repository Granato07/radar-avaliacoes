/**
 * Compara o modelo clássico com o LLM em avaliações reais do conjunto de teste.
 * Uso: GEMINI_API_KEY=... npm run comparar -- 60
 * Gera ../ml/comparacao.json (lido pelo README e pelo dashboard).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { classificar } from '../src/classico.js';
import { analisarComLLM } from '../src/llm.js';

const N = Number(process.argv[2] ?? 60);
if (!process.env.GEMINI_API_KEY) {
  console.error('Defina GEMINI_API_KEY para rodar a comparação.');
  process.exit(1);
}

// CSV gerado pelo treinar.py: texto,review_score,sentimento
function lerCsv(conteudo: string): string[][] {
  const linhas: string[][] = [];
  let campo = '', linha: string[] = [], aspas = false;
  for (let i = 0; i < conteudo.length; i++) {
    const ch = conteudo[i];
    if (aspas) {
      if (ch === '"' && conteudo[i + 1] === '"') { campo += '"'; i++; }
      else if (ch === '"') aspas = false;
      else campo += ch;
    } else if (ch === '"') aspas = true;
    else if (ch === ',') { linha.push(campo); campo = ''; }
    else if (ch === '\n') { linha.push(campo); linhas.push(linha); linha = []; campo = ''; }
    else if (ch !== '\r') campo += ch;
  }
  if (campo || linha.length) linhas.push([...linha, campo]);
  return linhas;
}
const linhas = lerCsv(readFileSync(new URL('../../data/teste.csv', import.meta.url), 'utf8'))
  .slice(1)
  .map(([texto, , real]) => ({ texto, real }));

// amostra estratificada: mesma quantidade de cada classe, para o neutro não sumir
const porClasse = ['negativo', 'neutro', 'positivo'].map((c) => linhas.filter((l) => l.real === c).slice(0, Math.ceil(N / 3)));
const amostra = porClasse.flat();

const res = { classico: { acertos: 0, ms: 0 }, llm: { acertos: 0, ms: 0, erros: 0 }, neutro: { classico: 0, llm: 0, total: 0 } };
const categorias: Record<string, number> = {};

for (const [i, { texto, real }] of amostra.entries()) {
  let t = performance.now();
  const c = classificar(texto);
  res.classico.ms += performance.now() - t;
  if (c.sentimento === real) res.classico.acertos++;

  t = performance.now();
  try {
    const l = await analisarComLLM(texto);
    res.llm.ms += performance.now() - t;
    if (l.sentimento === real) res.llm.acertos++;
    categorias[l.categoria] = (categorias[l.categoria] ?? 0) + 1;
    if (real === 'neutro') {
      res.neutro.total++;
      if (c.sentimento === 'neutro') res.neutro.classico++;
      if (l.sentimento === 'neutro') res.neutro.llm++;
    }
  } catch (e) {
    res.llm.erros++;
    console.warn(`#${i}: ${(e as Error).message}`);
  }
  await new Promise((r) => setTimeout(r, 400)); // respeita o limite gratuito
  process.stdout.write(`\r${i + 1}/${amostra.length}`);
}

const validas = amostra.length - res.llm.erros;
const saida = {
  amostras: amostra.length,
  classico: { acuracia: +(res.classico.acertos / amostra.length).toFixed(3), ms_medio: +(res.classico.ms / amostra.length).toFixed(2) },
  llm: { acuracia: +(res.llm.acertos / validas).toFixed(3), ms_medio: Math.round(res.llm.ms / validas), erros: res.llm.erros },
  recall_neutro: {
    classico: +(res.neutro.classico / res.neutro.total).toFixed(3),
    llm: +(res.neutro.llm / res.neutro.total).toFixed(3),
  },
  categorias_llm: categorias,
};
writeFileSync(new URL('../../ml/comparacao.json', import.meta.url), JSON.stringify(saida, null, 2));
console.log('\n', saida);
