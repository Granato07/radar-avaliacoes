import { describe, expect, it } from 'vitest';
import { createRequire } from 'node:module';
import { classificar } from '../src/classico.js';
import { normalizar, tokenizar } from '../src/texto.js';

const referencia = createRequire(import.meta.url)('./referencia-python.json') as {
  texto: string;
  probabilidades: Record<string, number>;
}[];

describe('modelo clássico', () => {
  it('normaliza igual ao Python (minúsculas e sem acento)', () => {
    expect(normalizar('  Ótimo   PRODUTO, chegou rápido! ')).toBe('otimo produto, chegou rapido!');
    expect(tokenizar('Não chegou: é 1 absurdo')).toEqual(['nao', 'chegou', 'absurdo']);
  });

  it('dá as mesmas probabilidades que o scikit-learn em 200 avaliações reais', () => {
    let maiorDiferenca = 0;
    for (const { texto, probabilidades } of referencia) {
      const r = classificar(texto);
      for (const [classe, p] of Object.entries(probabilidades)) {
        maiorDiferenca = Math.max(maiorDiferenca, Math.abs(r.probabilidades[classe as 'neutro'] - p));
      }
    }
    expect(maiorDiferenca).toBeLessThan(0.001);
  });

  it('classifica casos óbvios', () => {
    expect(classificar('Produto excelente, chegou antes do prazo. Recomendo!').sentimento).toBe('positivo');
    expect(classificar('Não recebi o produto e ninguém responde. Péssimo.').sentimento).toBe('negativo');
  });
});
