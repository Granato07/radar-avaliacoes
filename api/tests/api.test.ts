import { afterEach, describe, expect, it, vi } from 'vitest';
import request from 'supertest';
import { criarApp } from '../src/app.js';

const respostaGemini = (dados: unknown, status = 200) =>
  vi.fn(async () =>
    new Response(JSON.stringify({ candidates: [{ content: { parts: [{ text: JSON.stringify(dados) }] } }] }), { status }),
  ) as unknown as typeof fetch;

const llmValido = {
  sentimento: 'negativo',
  categoria: 'atraso',
  urgencia: 'alta',
  resumo: 'Pedido não entregue no prazo.',
  resposta_sugerida: 'Sentimos muito pelo atraso. Vamos verificar seu pedido agora.',
};

afterEach(() => {
  delete process.env.GEMINI_API_KEY;
});

describe('POST /api/analisar', () => {
  it('modo clássico funciona sem chave do LLM', async () => {
    const res = await request(criarApp()).post('/api/analisar').send({ texto: 'Chegou quebrado, péssimo', modo: 'classico' });
    expect(res.status).toBe(200);
    expect(res.body.classico.sentimento).toBe('negativo');
    expect(res.body.llm).toBeUndefined();
  });

  it('valida a entrada (400)', async () => {
    const app = criarApp();
    expect((await request(app).post('/api/analisar').send({ texto: 'a' })).status).toBe(400);
    expect((await request(app).post('/api/analisar').send({ texto: 'x'.repeat(1001) })).status).toBe(400);
    expect((await request(app).post('/api/analisar').send({ texto: 'ok ok ok', modo: 'outro' })).status).toBe(400);
  });

  it('modo llm sem chave responde 503', async () => {
    const res = await request(criarApp()).post('/api/analisar').send({ texto: 'Ainda não recebi', modo: 'llm' });
    expect(res.status).toBe(503);
  });

  it('modo ambos devolve as duas análises (LLM simulado)', async () => {
    process.env.GEMINI_API_KEY = 'chave-de-teste';
    const fetcher = respostaGemini(llmValido);
    const res = await request(criarApp(fetcher)).post('/api/analisar').send({ texto: 'Comprei há 20 dias e não chegou' });
    expect(res.status).toBe(200);
    expect(res.body.classico.sentimento).toBeDefined();
    expect(res.body.llm.categoria).toBe('atraso');
    expect(fetcher).toHaveBeenCalledOnce();
  });

  it('rejeita resposta do LLM fora do formato', async () => {
    process.env.GEMINI_API_KEY = 'chave-de-teste';
    const res = await request(criarApp(respostaGemini({ sentimento: 'feliz' })))
      .post('/api/analisar')
      .send({ texto: 'Gostei bastante', modo: 'llm' });
    expect(res.status).toBe(502);
  });

  it('repassa limite do Gemini como 429', async () => {
    process.env.GEMINI_API_KEY = 'chave-de-teste';
    const res = await request(criarApp(respostaGemini({}, 429))).post('/api/analisar').send({ texto: 'Gostei bastante', modo: 'llm' });
    expect(res.status).toBe(429);
  });

  it('aplica limite de requisições por IP', async () => {
    const app = criarApp();
    let ultimo = 200;
    for (let i = 0; i < 25; i++) ultimo = (await request(app).post('/api/analisar').send({ texto: 'teste de limite', modo: 'classico' })).status;
    expect(ultimo).toBe(429);
  });
});

describe('outras rotas', () => {
  it('GET /api/saude e /api/metricas', async () => {
    const app = criarApp();
    expect((await request(app).get('/api/saude')).body.ok).toBe(true);
    expect((await request(app).get('/api/metricas')).body.acuracia).toBeGreaterThan(0.7);
    expect((await request(app).get('/nada')).status).toBe(404);
  });
});
