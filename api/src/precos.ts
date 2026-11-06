// Preço por 1 milhão de tokens (US$), em chamadas normais (sem desconto de batch).
// Fontes consultadas em 06/11/2026: tabela do GroqCloud (out/2026) e preços públicos do Claude.
// Confira a página oficial de cada provedor antes de usar os números para orçar.
export const PRECOS_POR_MILHAO: Record<string, { entrada: number; saida: number }> = {
  'openai/gpt-oss-20b': { entrada: 0.075, saida: 0.3 },
  'openai/gpt-oss-120b': { entrada: 0.15, saida: 0.6 },
  'claude-haiku-5-5': { entrada: 0.1, saida: 0.5 },
  'claude-sonnet-5-5': { entrada: 2, saida: 10 },
  'claude-opus-5-5': { entrada: 4, saida: 20 },
};

export interface Uso {
  tokens_entrada: number;
  tokens_saida: number;
  custo_usd: number | null; // null quando o modelo não está na tabela
}

export function calcularUso(modelo: string, entrada: number, saida: number): Uso {
  const preco = PRECOS_POR_MILHAO[modelo];
  const custo = preco ? (entrada * preco.entrada + saida * preco.saida) / 1_000_000 : null;
  return { tokens_entrada: entrada, tokens_saida: saida, custo_usd: custo === null ? null : Number(custo.toFixed(8)) };
}
