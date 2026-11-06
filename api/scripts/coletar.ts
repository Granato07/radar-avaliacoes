// Coleta as respostas da API em produção para a amostra avaliada pelo juiz.
// Uso: API_URL=https://radar-avaliacoes-api.vercel.app npm run coletar
// Salva ml/juiz/respostas.json. A análise fica em ml/juiz/analisar.py.
import { readFileSync, writeFileSync } from 'node:fs';

const API = process.env.API_URL ?? 'http://localhost:3333';
const MODELOS = ['openai/gpt-oss-20b', 'openai/gpt-oss-120b'];
const ESPERA_MS = 4500; // respeita o limite por IP da API e a cota do Groq

function lerCsv(caminho: string): Record<string, string>[] {
  const txt = readFileSync(caminho, 'utf8');
  const linhas: string[][] = [];
  let campo = '', linha: string[] = [], aspas = false;
  for (let i = 0; i < txt.length; i++) {
    const c = txt[i];
    if (aspas) {
      if (c === '"' && txt[i + 1] === '"') { campo += '"'; i++; } else if (c === '"') aspas = false; else campo += c;
    } else if (c === '"') aspas = true;
    else if (c === ',') { linha.push(campo); campo = ''; }
    else if (c === '\n') { linha.push(campo); linhas.push(linha); linha = []; campo = ''; }
    else campo += c;
  }
  if (campo || linha.length) { linha.push(campo); linhas.push(linha); }
  const [cab, ...dados] = linhas;
  return dados.map((l) => Object.fromEntries(cab.map((k, i) => [k, l[i]])));
}

const amostra = lerCsv(new URL('../../ml/juiz/amostra.csv', import.meta.url).pathname);
const respostas: Record<string, unknown>[] = [];
for (const modelo of MODELOS) {
  for (const { id, texto } of amostra) {
    for (let tentativa = 1; tentativa <= 3; tentativa++) {
      const r = await fetch(`${API}/api/analisar`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texto, modo: 'ambos', modelo }),
      });
      const corpo = (await r.json()) as { classico?: unknown; llm?: { erro?: string } };
      await new Promise((ok) => setTimeout(ok, ESPERA_MS));
      if (r.ok && corpo.llm && !corpo.llm.erro) { respostas.push({ id: Number(id), modelo, ...corpo }); break; }
      if (tentativa === 3) respostas.push({ id: Number(id), modelo, erro: corpo.llm?.erro ?? r.status });
    }
    process.stdout.write('.');
  }
}
writeFileSync(new URL('../../ml/juiz/respostas.json', import.meta.url).pathname, JSON.stringify(respostas, null, 1));
console.log(`\n${respostas.length} respostas salvas`);
