
process.env.TF_CPP_MIN_LOG_LEVEL = '2';      // silencia INFO/WARN
process.env.TF_ENABLE_ONEDNN_OPTS = '0';     // desativa avisos do oneDNN

// use dynamic import to apply env flags before TensorFlow initializes
const tfModule = await import('@tensorflow/tfjs-node');
const tf = tfModule.default ?? tfModule;

/*
async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    // Primeira camada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizacoes)

    // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e consequentemente, mais processamento ela vai usar

    // A ReLU age como um filtro:
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    /// Se a informação chegou nesse neuronio é positiva, passa para frente!
    // se for zero ou negativa, pode jogar fora, nao vai servir para nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // Saída: 3 neuronios
    // um para cada categoria (premium, medium, basic)

    // activation: softmax normaliza a saida em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // Compilando o modelo
    // optimizer Adam ( Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com historico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium será sempre [1, 0, 0]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendação, categorização de
    // usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    // verbose: desabilita o log interno (e usa só callback)
    // epochs: quantidade de veses que vai rodar no dataset
    // shuffle: embaralha os dados, para evitar viés
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                //  onEpochEnd: (epoch, log) => console.log(
                //      `Epoch: ${epoch}: loss = ${log.loss}`
                //  )
            }
        }
    )

    return model
}

async function predict(model, pessoa) {
    // transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dado melhor!
// assim o algoritmo consegue entender melhor os padrões complexos
// dos dados
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: "Curitiba" }
// Normalizando a idade da nova pessoa usando o mesmo padrão do treino
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25 ) = 0.2

const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        1,    // cor azul
        0,    // cor vermelho
        0,    // cor verde
        0,    // localização São Paulo
        1,    // localização Rio
        0     // localização Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results) 

*/

//**************************************************************** */



async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()
    // idade + salario + 3 cores + 3 localizações + 4 profissões + 4 escolaridades = 16
    model.add(tf.layers.dense({ inputShape: [16], units: 80, activation: 'relu' }))
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] })
    await model.fit(inputXs, outputYs, { verbose: 0, epochs: 100, shuffle: true,callbacks: {} })
    return model
}

async function predict(model, pessoa) {
    // transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

// Novo exemplo de pessoas para treino (idade, cor, localização, salário, profissão, escolaridade)
// const pessoas = [
//     { nome: "Erick",   idade: 30, cor: "azul",     localizacao: "São Paulo", salario: 6500,  profissao: "programador",     escolaridade: "bacharelado" },
//     { nome: "Ana",     idade: 25, cor: "vermelho", localizacao: "Rio",       salario: 9500,  profissao: "analista",        escolaridade: "medio" },
//     { nome: "Carlos",  idade: 40, cor: "verde",    localizacao: "Curitiba",  salario: 11000, profissao: "eng_software",    escolaridade: "pos" },
//     { nome: "Beatriz", idade: 28, cor: "azul",     localizacao: "São Paulo", salario: 12000, profissao: "eng_ia_aplicada", escolaridade: "mestre" },
//     { nome: "Diego",   idade: 35, cor: "vermelho", localizacao: "Rio",       salario: 7500,  profissao: "programador",     escolaridade: "bacharelado" },
//     { nome: "Fernanda",idade: 32, cor: "verde",    localizacao: "Curitiba",  salario: 10200,  profissao: "analista",        escolaridade: "medio" },
//     { nome: "Gustavo", idade: 45, cor: "azul",     localizacao: "Rio",       salario: 9300,  profissao: "eng_software",    escolaridade: "pos" },
//     { nome: "Helena",  idade: 22, cor: "vermelho", localizacao: "São Paulo", salario: 4100,  profissao: "programador",     escolaridade: "medio" },
// ]

const tensorPessoasNormalizado = [
  [0.33, 0.40, 1,0,0, 1,0,0, 1,0,0,0, 0,1,0,0], // Erick
  [0.00, 0.20, 0,1,0, 0,1,0, 0,1,0,0, 1,0,0,0], // Ana
  [1.00, 0.80, 0,0,1, 0,0,1, 0,0,1,0, 0,0,1,0], // Carlos
  [0.20, 1.00, 1,0,0, 1,0,0, 0,0,0,1, 0,0,0,1], // Beatriz
  [0.67, 0.50, 0,1,0, 0,1,0, 1,0,0,0, 0,1,0,0], // Diego
  [0.47, 0.68, 0,0,1, 0,0,1, 0,1,0,0, 1,0,0,0], // Fernanda
  [1.20, 0.38, 1,0,0, 0,1,0, 0,0,1,0, 0,0,1,0], // Gustavo
  [-0.20, 0.00, 0,1,0, 1,0,0, 1,0,0,0, 1,0,0,0], // Helena
]

const labelsNomes = ["premium", "medium", "basic"];
const tensorLabels = [
  [1, 0, 0], // Erick    -> premium
  [0, 1, 0], // Ana      -> medium
  [0, 0, 1], // Carlos   -> basic
  [1, 0, 0], // Beatriz  -> premium
  [0, 1, 0], // Diego    -> medium
  [0, 1, 0], // Fernanda -> medium
  [0, 0, 1], // Gustavo  -> basic
  [0, 1, 0], // Helena   -> medium
]


const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)


const pessoa = {
  nome: 'zé',
  idade: 28,
  cor: 'verde',
  localizacao: 'Curitiba',
  salario: 8200,
  profissao: 'eng_ia_aplicada',
  escolaridade: 'pos'
}

// Normalização exemplo (idade_min=22, idade_max=45; salario_min=4100, salario_max=12000)
const pessoaTensorNormalizado = [
  [
    0.55, // idade_norm = (28-22)/(45-22)
    0.80, // salario_norm = (8200-4100)/(12000-4100)
    0,0,1, // cor verde
    0,0,1, // Curitiba
    0,0,0,1, // prof_eng_ia_aplicada
    1,0,0,0  // esc_pos
  ]
]

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results) 