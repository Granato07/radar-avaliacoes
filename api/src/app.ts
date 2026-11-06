import express, { type NextFunction, type Request, type Response } from 'express';
import { z } from 'zod';
import { classificar } from './classico.js';
import { analisarComLLM, ErroLLM, llmConfigurado, modelosDisponiveis } from './llm.js';
import metricas from './modelo/metricas.json' with { type: 'json' };

export const esquemaEntrada = z.object({
  texto: z.string().trim().min(3, 'texto muito curto').max(1000, 'texto muito longo (máx. 1000 caracteres)'),
  modo: z.enum(['classico', 'llm', 'ambos', 'triagem']).default('ambos'),
  modelo: z.string().max(80).optional(),
});

// Triagem: o modelo clássico decide sozinho quando está muito confiante de que a avaliação é positiva
// (elogio não precisa de categoria, urgência nem resposta). O resto vai para o LLM.
// O limiar saiu da avaliação em ml/juiz (ver README).
const LIMIAR_TRIAGEM = Number(process.env.LIMIAR_TRIAGEM ?? 0.8);

// Limite simples por IP para proteger a cota do LLM (em memória, por instância)
const LIMITE = Number(process.env.LIMITE_POR_MINUTO ?? 20);
function criarLimitador() {
  const acessos = new Map<string, number[]>();
  return (req: Request, res: Response, next: NextFunction) => {
    const ip = req.ip ?? 'desconhecido';
    const agora = Date.now();
    const recentes = (acessos.get(ip) ?? []).filter((t) => agora - t < 60_000);
    if (recentes.length >= LIMITE) {
      res.status(429).json({ erro: 'Muitas requisições. Tente de novo em 1 minuto.' });
      return;
    }
    acessos.set(ip, [...recentes, agora]);
    next();
  };
}

export function criarApp(fetcher: typeof fetch = fetch) {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', 1);
  app.use(express.json({ limit: '8kb' }));

  // CORS: só o painel (e localhost em desenvolvimento) pode chamar a API pelo navegador
  const origens = (process.env.ORIGENS_PERMITIDAS ?? 'http://localhost:3000').split(',').map((o) => o.trim());
  app.use((req, res, next) => {
    const origem = req.headers.origin;
    if (origem && origens.includes(origem)) {
      res.setHeader('Access-Control-Allow-Origin', origem);
      res.setHeader('Vary', 'Origin');
      res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    }
    if (req.method === 'OPTIONS') {
      res.sendStatus(204);
      return;
    }
    next();
  });

  app.get('/api/saude', (_req, res) => {
    res.json({ ok: true, llm_configurado: llmConfigurado(), modelos: modelosDisponiveis(), limiar_triagem: LIMIAR_TRIAGEM });
  });

  app.get('/api/metricas', (_req, res) => {
    res.json(metricas);
  });

  app.post('/api/analisar', criarLimitador(), async (req, res) => {
    const entrada = esquemaEntrada.safeParse(req.body);
    if (!entrada.success) {
      res.status(400).json({ erro: 'Entrada inválida', detalhes: entrada.error.issues.map((i) => i.message) });
      return;
    }
    const { texto, modo, modelo } = entrada.data;
    const resposta: Record<string, unknown> = { texto };

    let classico: ReturnType<typeof classificar> | undefined;
    if (modo !== 'llm') {
      const inicio = performance.now();
      classico = classificar(texto);
      resposta.classico = { ...classico, tempo_ms: Math.round(performance.now() - inicio) };
    }

    let chamarLLM = modo !== 'classico';
    if (modo === 'triagem' && classico) {
      const confiantePositivo = classico.sentimento === 'positivo' && classico.confianca >= LIMIAR_TRIAGEM;
      chamarLLM = !confiantePositivo;
      resposta.triagem = {
        llm_acionado: chamarLLM,
        limiar: LIMIAR_TRIAGEM,
        motivo: confiantePositivo
          ? 'modelo clássico confiante de que é positiva: LLM não foi chamado'
          : 'não é positiva com confiança: LLM chamado para categoria, urgência e resposta',
      };
    }

    if (chamarLLM) {
      const inicio = performance.now();
      try {
        resposta.llm = { ...(await analisarComLLM(texto, fetcher, modelo)), tempo_ms: Math.round(performance.now() - inicio) };
      } catch (e) {
        const erro = e instanceof ErroLLM ? e : new ErroLLM('Falha ao consultar o LLM');
        if (modo === 'llm') {
          res.status(erro.status).json({ erro: erro.message });
          return;
        }
        resposta.llm = { erro: erro.message };
      }
    }
    res.json(resposta);
  });

  app.use((_req, res) => {
    res.status(404).json({ erro: 'Rota não encontrada' });
  });
  return app;
}

export default criarApp();
