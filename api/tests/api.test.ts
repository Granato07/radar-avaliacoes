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
  delete process.env.GROQ_API_KEY;
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

describe('LLM com modelos reserva', () => {
  it('usa o próximo modelo quando o primeiro falha', async () => {
    process.env.GEMINI_API_KEY = 'chave-de-teste';
    const ok = { candidates: [{ content: { parts: [{ text: JSON.stringify(llmValido) }] } }] };
    let chamada = 0;
    const fetcher = vi.fn(async () => (++chamada === 1 ? new Response('', { status: 503 }) : new Response(JSON.stringify(ok)))) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher)).post('/api/analisar').send({ texto: 'Não chegou ainda', modo: 'llm' });
    expect(res.status).toBe(200);
    expect(res.body.llm?.modelo ?? res.body.modelo).toBeDefined();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('timeout vira 504 com mensagem clara', async () => {
    process.env.GEMINI_API_KEY = 'chave-de-teste';
    const fetcher = vi.fn(async () => {
      throw Object.assign(new Error('timeout'), { name: 'TimeoutError' });
    }) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher)).post('/api/analisar').send({ texto: 'Não chegou ainda', modo: 'llm' });
    expect(res.status).toBe(504);
    expect(res.body.erro).toMatch(/demorou/);
  });
});

describe('Groq como provedor principal', () => {
  const respostaGroq = (dados: unknown) => new Response(JSON.stringify({ choices: [{ message: { content: JSON.stringify(dados) } }] }));

  it('usa o Groq com schema strict e devolve provedor/modelo', async () => {
    process.env.GROQ_API_KEY = 'chave-groq';
    const fetcher = vi.fn(async () => respostaGroq(llmValido)) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher)).post('/api/analisar').send({ texto: 'Não chegou ainda', modo: 'llm' });
    expect(res.status).toBe(200);
    expect(res.body.llm.provedor).toBe('groq');
    expect(res.body.llm.modelo).toBe('openai/gpt-oss-20b');
    const [url, init] = (fetcher as unknown as { mock: { calls: [string, RequestInit][] } }).mock.calls[0];
    expect(url).toContain('api.groq.com');
    const corpo = JSON.parse(init.body as string);
    expect(corpo.response_format.json_schema.strict).toBe(true);
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer chave-groq');
  });

  it('cai para o Gemini quando o Groq falha', async () => {
    process.env.GROQ_API_KEY = 'chave-groq';
    process.env.GEMINI_API_KEY = 'chave-gemini';
    const fetcher = vi.fn(async (url: string) =>
      url.includes('groq')
        ? new Response('', { status: 503 })
        : new Response(JSON.stringify({ candidates: [{ content: { parts: [{ text: JSON.stringify(llmValido) }] } }] })),
    ) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher)).post('/api/analisar').send({ texto: 'Não chegou ainda', modo: 'llm' });
    expect(res.status).toBe(200);
    expect(res.body.llm.provedor).toBe('gemini');
  });

  it('chave inválida pula os outros modelos do mesmo provedor', async () => {
    process.env.GROQ_API_KEY = 'chave-errada';
    const fetcher = vi.fn(async () => new Response('', { status: 401 })) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher)).post('/api/analisar').send({ texto: 'Não chegou ainda', modo: 'llm' });
    expect(res.status).toBe(502);
    expect(fetcher).toHaveBeenCalledOnce();
  });
});

describe('tokens, custo, escolha de modelo e triagem', () => {
  const groqComUso = (dados: unknown) =>
    new Response(
      JSON.stringify({ choices: [{ message: { content: JSON.stringify(dados) } }], usage: { prompt_tokens: 400, completion_tokens: 100 } }),
    );

  it('devolve tokens e custo calculado pela tabela de preços', async () => {
    process.env.GROQ_API_KEY = 'chave-groq';
    const fetcher = vi.fn(async () => groqComUso(llmValido)) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher)).post('/api/analisar').send({ texto: 'Não chegou ainda', modo: 'llm' });
    expect(res.body.llm.uso).toEqual({ tokens_entrada: 400, tokens_saida: 100, custo_usd: 0.00006 }); // 400*0,075 + 100*0,30 por milhão
  });

  it('modelo escolhido é usado sozinho, sem cair para outro', async () => {
    process.env.GROQ_API_KEY = 'chave-groq';
    const fetcher = vi.fn(async () => new Response('', { status: 503 })) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher))
      .post('/api/analisar')
      .send({ texto: 'Não chegou ainda', modo: 'llm', modelo: 'openai/gpt-oss-120b' });
    expect(res.status).toBe(502);
    expect(fetcher).toHaveBeenCalledOnce();
    const corpo = JSON.parse((fetcher as unknown as { mock: { calls: [string, RequestInit][] } }).mock.calls[0][1].body as string);
    expect(corpo.model).toBe('openai/gpt-oss-120b');
  });

  it('recusa modelo que não está configurado (400)', async () => {
    process.env.GROQ_API_KEY = 'chave-groq';
    const res = await request(criarApp()).post('/api/analisar').send({ texto: 'Não chegou ainda', modo: 'llm', modelo: 'modelo-caro' });
    expect(res.status).toBe(400);
  });

  it('triagem: elogio claro não chama o LLM', async () => {
    process.env.GROQ_API_KEY = 'chave-groq';
    const fetcher = vi.fn(async () => groqComUso(llmValido)) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher))
      .post('/api/analisar')
      .send({ texto: 'Excelente produto, chegou antes do prazo. Recomendo!', modo: 'triagem' });
    expect(res.body.triagem.llm_acionado).toBe(false);
    expect(res.body.llm).toBeUndefined();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('triagem: reclamação vai para o LLM', async () => {
    process.env.GROQ_API_KEY = 'chave-groq';
    const fetcher = vi.fn(async () => groqComUso(llmValido)) as unknown as typeof fetch;
    const res = await request(criarApp(fetcher)).post('/api/analisar').send({ texto: 'Paguei e não recebi o produto', modo: 'triagem' });
    expect(res.body.triagem.llm_acionado).toBe(true);
    expect(res.body.llm.categoria).toBe('atraso');
  });
});
