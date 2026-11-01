import app from './app.js';

const porta = Number(process.env.PORT ?? 3333);
app.listen(porta, () => console.log(`Radar de Avaliações API em http://localhost:${porta}`));
