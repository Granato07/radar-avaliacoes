// Pré-processamento idêntico ao do treino em Python (ml/treinar.py):
// minúsculas, sem acentos, tokens de 2+ letras/números, unigramas e bigramas.

export function normalizar(texto: string): string {
  return texto
    .toLowerCase()
    .normalize('NFD')
    .replace(/\p{Mn}/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export function tokenizar(texto: string): string[] {
  return normalizar(texto).match(/[\p{L}\p{N}_]{2,}/gu) ?? [];
}

export function ngramas(tokens: string[], min = 1, max = 2): string[] {
  const saida: string[] = [];
  for (let n = min; n <= max; n++) {
    for (let i = 0; i + n <= tokens.length; i++) saida.push(tokens.slice(i, i + n).join(' '));
  }
  return saida;
}
