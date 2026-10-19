# Documento de Rastreio — OpenAPI Rules Bank

Este documento é um caderno de bordo do projeto. A ideia é registrar, em texto
corrido, as ideias que motivaram cada etapa, o que foi de fato implementado, o
que foi mudado e por quê, e o que foi testado — de forma que sirva como matéria-prima
para a escrita da dissertação de mestrado mais adiante. Não pretende ser documentação
técnica (isso está no README e no código); pretende contar a *história* das decisões.

Ele não é versionado (está no `.gitignore`), justamente por ser um registro pessoal e
em evolução. A cada mudança relevante no projeto, acrescente um parágrafo novo em vez
de reescrever o que já está aqui.

Última atualização: 2026-09-30 (parágrafos de ago/set reconstruídos — ver nota abaixo).

---

## A ideia central

O objetivo do projeto é transformar uma especificação técnica normativa do 3GPP
(por exemplo, a TS 28.532) em um "banco de regras" (rules bank) estruturado, que
possa depois orientar a geração de APIs OpenAPI compatíveis com o 3GPP. A intuição
é que grande parte do trabalho de escrever uma OpenAPI conforme uma spec 3GPP é, na
prática, um trabalho de leitura e tradução: entender o que o texto normativo exige e
mapear cada exigência para um elemento concreto de uma OpenAPI (um caminho, um método,
uma propriedade de schema, uma resposta, um parâmetro). O projeto tenta automatizar
esse trabalho de tradução usando um pipeline de múltiplos agentes de LLM.

A escolha por múltiplos agentes, e não um único prompt grande, veio da percepção de
que a tarefa tem etapas de naturezas diferentes — planejar o que extrair, extrair,
questionar criticamente o que foi extraído, validar e consolidar — e que separar essas
responsabilidades torna cada etapa mais controlável, mais fácil de depurar e mais
auditável. Cada agente é um nó de um grafo (implementado com LangGraph), e o estado
compartilhado flui entre eles.

Uma segunda decisão estrutural importante foi processar o documento **seção por seção**,
com um laço de refinamento em cada seção, em vez de processar o documento inteiro de
uma vez. Isso mantém os erros localizados, permite refinar apenas o que falhou e evita
reprocessar tudo quando só um trecho precisa de correção.

---

## Como o pipeline funciona hoje

O fluxo começa por um leitor (Reader) que carrega o documento 3GPP principal — já
convertido de `.docx` para Markdown numa etapa de pré-processamento — e o divide em
seções. Também carrega documentos auxiliares opcionais, um YAML de release anterior
(quando se está construindo regras para uma release mais nova) e a referência da
especificação OpenAPI.

Em seguida vem um construtor de contrato (ContractBuilder), o nó mais recente do
pipeline. Ele constrói, apenas a partir do texto da spec, um "contrato de escopo" do
serviço-alvo: o que o serviço faz, quais são suas capacidades, seus conceitos-chave e
quais termos indicam que uma regra pertence — ou não pertence — a ele. Esse contrato é
depois injetado no planejador e no extrator para que ambos saibam o que está dentro e o
que está fora do escopo do serviço.

O planejador (Planner) lê os títulos e prévias das seções e decide o que extrair de
cada uma, com prioridade e foco. O extrator (Extractor) processa a seção corrente e
produz regras brutas, cada uma associada a um trecho normativo e a um mapeamento para
um elemento OpenAPI. O refletor (Reflector) faz uma auto-crítica: por regra, avalia se
ela está bem fundamentada no texto e se o mapeamento está correto, atribuindo uma
confiança; por seção, analisa o que ainda estaria faltando. O validador (Validator) faz
a verificação estrutural (tipos e consistência do mapeamento) e depois uma validação
semântica em estágios, decidindo o que é válido, o que precisa de correção, o que deve
ser dividido e o que deve ser descartado. Por fim, o construtor (Builder) monta e salva
o banco de regras em JSON depois de cada seção.

O laço de feedback tem uma característica de projeto deliberada: o refletor *analisa*,
mas quem *decide* é o validador, e o extrator só age no que o validador manda. Essa
separação foi introduzida para evitar que o extrator reagisse a sugestões ainda não
confirmadas — o validador é a única fonte de instruções de correção.

---

## Histórico das decisões e mudanças

**Início (março de 2026).** O projeto começou com a estrutura básica do grafo e os
primeiros códigos do fluxo. Logo em seguida veio a montagem do RAG (indexação da
referência OpenAPI no Qdrant) — a ideia era dar aos agentes um contexto de consulta
sobre a própria especificação OpenAPI, para ancorar os mapeamentos. Ainda em março
foram implementados, um a um, os nós de extração, reflexão, validação e construção, cada
um testado isoladamente. Já nessa fase apareceu a necessidade de tratar corretamente o
comportamento de "loop-back" (voltar a uma seção para corrigi-la), que foi ajustado no
validador e no extrator.

**Redesenho do laço de feedback (abril de 2026).** A primeira versão do laço se mostrou
insuficiente, e houve um redesenho amplo: schemas, prompts e estado foram retrabalhados
para separar melhor os canais de feedback. Nesse período também entrou o coletor de
diagnóstico (DiagnosticCollector), motivado pela necessidade de enxergar o que acontecia
a cada iteração — sem essa telemetria era difícil entender por que uma seção repetia ou
por que uma regra falhava. A mudança mais significativa foi transformar a validação
semântica em um processo de três estágios (inicialização da seção, veredito por regra e
revisão da reflexão), tornando a decisão do validador mais explícita e auditável. Os
prompts do extrator e do planejador foram melhorados em seguida, com base no que os
diagnósticos revelaram sobre a qualidade das extrações.

**Empacotamento e entradas reais (maio–junho de 2026).** O código foi reorganizado para
um pacote instalável, e os runs de documento completo passaram a ser salvos separadamente
dos parciais. A grande mudança de junho foi lidar com as entradas reais do 3GPP: as specs
vêm em `.docx`, então foi criada uma etapa de pré-processamento que as converte para
Markdown com o pandoc antes do pipeline rodar. Também passou a ser possível escolher qual
documento OpenAPI do Annex A seria o alvo e fornecer um YAML de release anterior como
contexto. A referência OpenAPI passou a ser buscada por versão configurada — a versão
3.0.1 foi escolhida por ser a que casa com os inputs 3GPP, mantendo prompts e validação
genéricos quanto à versão. O argumento `--target` foi tornado obrigatório, porque a
escolha do serviço-alvo é essencial para dar foco à extração. No fim de junho foi
adicionado um helper de estado inicial e um checkpointer opcional no grafo, pensando em
acionar o pipeline a partir de um host externo (uma UI de chatbot) com o mesmo estado
inicial da linha de comando.

**Contrato de escopo do serviço (julho de 2026).** A mudança conceitual mais recente foi
abandonar o uso do Annex A (o YAML embutido na spec) como contexto de extração. A ideia
anterior era extrair a subseção do YAML alvo do Annex A e injetá-la nos prompts; isso foi
descartado em favor de construir as regras **apenas a partir das seções normativas** da
spec. O raciocínio: o YAML do Annex A é a *resposta pronta*, e usá-lo como contexto
arriscava fazer o pipeline copiar a solução em vez de derivá-la do texto normativo — o
que enfraqueceria a tese de que é possível gerar regras a partir do texto. No lugar do
Annex A entrou o ContractBuilder, que constrói um entendimento explícito e enxuto do
escopo do serviço a partir do próprio texto e o usa para filtrar regras fora de contexto
(por exemplo, reconhecer que uma menção a alarme pertence a outro serviço de gerência, e
não ao serviço-alvo), sem despejar o YAML inteiro no prompt.

**Nota de recuperação (30/set).** Os parágrafos de agosto e setembro abaixo foram
RECONSTRUÍDOS em 30/set a partir do histórico da conversa, das memórias do projeto e dos
arquivos Problems_ProvMnS: em 28/set uma ferramenta externa sobrescreveu este documento
(não versionado) com a versão de julho, apagando os registros originais destas sessões. O
código correspondente não se perdeu (está nos commits do git). Detalhes em
[memória: 28set-revert-event].

**Qualidade de regras vs. o YAML oficial (agosto de 2026).** Validando o pipeline contra o
YAML oficial da PerfMnS, uma sessão longa expôs uma família de defeitos e a lição central
de que, onde o pipeline errava de forma sistemática, a causa era quase sempre uma regra
genérica demais ou um contexto ausente numa fase que decide — não uma falha pontual do
modelo. Consertos daquela sessão: (a) o cliente LLM ganhou timeout e retry (uma chamada
pendurada travava o run por minutos); (b) a seção do POST passou a ser classificada como
recurso, não mapeamento, pelo critério de ter tabela de respostas — o que destravou as
respostas 204/4XX/5XX que antes se perdiam; (c) a fase de escopo (Fase 3) deixou de
descartar as próprias respostas da seção, exigindo evidência literal nas regras já criadas;
(d) o par 4xx/5xx deixou de ser fundido ou barrado por caixa (prompt enfático + normalização
determinística); (e) split deixou de matar uma regra atômica válida (um 5XX não é split de
um 4XX). O modelo migrou para gpt-5.6-luna via OpenRouter, com uma melhora clara.

**Dar contexto às fases que decidem (agosto/setembro).** A lição acima virou método. O
`reflection_confidence` foi redefinido como `reflector_score` — mede se a regra deve estar
no conjunto final, não a fidelidade ao texto — e passou a acionar o descarte. O veredito da
pré-análise (campos herdados via allOf, dono do schema) passou a ser propagado às três
fases que calculam regras faltantes (Reflector fase 2, Validator estágios 1 e 3), para que
um campo herdado nunca fosse pedido como faltante; e as fases por-regra passaram a ver as
regras irmãs da seção, para não alucinar que falta um irmão já extraído. Consolidou-se que
cada alvo recebe UMA classificação — correção OU faltante, nunca as duas. Resultado: os
campos herdados de NotificationHeader deixaram de vazar, e o enum PerfMetricDirection voltou
a bater o YAML oficial. Também se descobriu que "todas as regras válidas" não é "todas as
regras presentes": o request_body do POST podia sumir sem deixar marca de erro — é preciso
medir cobertura contra o ground truth (ver [memória: request_body_lost-562]).

**Notificação é callback só quando o consumidor hospeda o endpoint (setembro).** A regra
absoluta "notificação é callback, nunca path" reprovava o path_operation legítimo do
endpoint que o próprio serviço expõe (o POST no /notificationSink). Trocou-se por um critério
genérico: callback quando o destino é um endereço que o consumidor fornece em runtime; path
quando é um recurso fixo que o serviço hospeda. Sem citar caminhos concretos.

**Referência externa: a regressão do ErrorResponse e a direção de "adiar para a seção
futura" (setembro).** Ao voltar ao PerfMnS, uma regressão apareceu: o `$ref` da resposta de
erro perdeu o nome do arquivo YAML da spec onde o tipo é publicado — saía
"#/components/schemas/ErrorResponse" (interno) em vez de
"TS28623_ComDefs.yaml#/…/ErrorResponse". A causa, medida e registrada em detalhe nos
arquivos Problems_ProvMnS (§3.23), foi uma mudança anterior — a separação entre "tipo
importado" e "spec apenas citada como contexto" — que tornou a fase 1 rigorosa: ela passou
a só registrar a origem quando uma mesma linha nomeia o tipo e a spec juntos. Isso eliminou
uma origem falsa (um tipo local atribuído a outra spec), mas eliminou também vínculos que a
fase 1 antes inventava e que por acaso acertavam — o ErrorResponse era um deles. E não é
caso de aceitar a perda: há gancho textual real (a tabela de tipos cita "TS 28.623 [44]"),
então a regra de vínculo é que não está reconhecendo uma tabela que a satisfaz — uma
hipótese é o número de referência "[44]" colado ao "TS 28.623" não casar com a forma limpa.
O impacto medido é grande: ~30 ocorrências, um quarto das regras do ProvMnS. A ordem (a
seção que usa o tipo roda antes da que o declara) agrava, mas não é a raiz. Construiu-se o mecanismo
de "adiar": a seção que usa, sem saber o arquivo, não emite a regra; a seção futura que
declara o tipo a emite com o nome do arquivo; nenhuma estrutura nova de estado, o tipo fica
na memória com arquivo vazio (pendente) e é atualizado quando conhecido. O mecanismo foi
construído e verificado em unidade, mas nos testes NÃO disparou: a fase 1 da seção que usa
nem reconhece o ErrorResponse como externo (a spec só é declarada depois), então o gatilho
do adiamento não ativa. Ficaram mapeadas duas soluções complementares — a fase 1 conseguir
marcar um tipo usado-sem-origem como externo-a-descobrir, e o ContractBuilder popular a
memória de imports cedo (mas ele só lê o preview das seções, que não alcança as declarações;
reordenar exigiria uma varredura completa, custosa). Detalhe em
[memória: errorresponse-ref-regression]. Também se alinhou, com o repo irmão
openapi_generator, seis divergências entre os dois e o YAML oficial — a principal do lado do
rulesbank sendo que o endereço de um query_parameter precisa carregar o método
(paths.<path>.<method>), senão o generator não sabe a que operação o parâmetro pertence.

---

## O que foi testado e o que se aprendeu

O projeto tem testes unitários rápidos (conversão de `.docx` e o parsing de seções, que
precisa lidar tanto com títulos em estilo ATX quanto setext, já que versões diferentes do
pandoc produzem formatos diferentes) e testes de integração que rodam o pipeline de fato
contra o LLM — um sobre o documento completo e outro com escopo reduzido a dez seções,
mais rápido e mais barato para iterar. Há também notebooks exploratórios por nó, que
foram o principal instrumento de teste manual durante o desenvolvimento de cada agente.

Ao longo do desenvolvimento, os artefatos de saída registrados mostram a evolução dos
experimentos: primeiro runs parciais (fim de março a meados de abril), depois runs de
documento completo (a partir de meados de abril), com os resultados "finais" arquivados
separadamente. Os relatórios de diagnóstico concentram-se no período de abril, quando o
laço de feedback estava sendo redesenhado e era mais importante observar o comportamento
iteração a iteração.

Os aprendizados que motivaram mudanças foram, em resumo: (1) um laço de feedback ingênuo
não basta — foi preciso separar análise de decisão para o pipeline convergir sem o
extrator "perseguir" sugestões instáveis; (2) sem telemetria por iteração é praticamente
impossível depurar um pipeline de agentes, o que justificou o diagnóstico; (3) usar a
resposta pronta (o YAML do Annex A) como contexto compromete a validade do experimento,
o que levou ao contrato de escopo derivado só do texto.

---

## Pendências e ideias em aberto

Há uma refatoração de schema ainda pendente (renomear os campos de reflexão e adicionar
um indicador explícito de validação à regra validada), adiada por não ser bloqueante.
Novas ideias e mudanças devem ser acrescentadas aqui conforme surgirem, sempre com o
registro do *porquê* — que é a parte que mais interessa para a dissertação.

---

## Erro de grafia na própria spec (3GPP TS 28.532) — `NotifyAttributeValueChanges`

Registro de um defeito que não é nosso: está no texto da 3GPP e dava erro no pipeline.

**Onde:** seção **12.1.1.3.2.4.3.1** (a operação POST do recurso de callback), na
**Tabela 12.1.1.3.2.4.3.1-2** ("Data structures supported by the POST Request Body on
this resource"). No arquivo-fonte que o pipeline lê,
`data/inputs/3gpp/md/28532-i20.md`, a tabela está na **linha 5964** (a chamada à tabela
está na linha 5951, o título na 5954).

**O defeito:** a coluna "Data type" dessa tabela, redigida à mão, grafa o mesmo conjunto
de tipos de três formas inconsistentes. Para o tipo das mudanças de atributo, escreve
**`NotifyAttributeValueChanges`** — **faltando o "Moi" no meio**. O nome canônico, usado
no heading da cláusula do tipo e no `components/schemas` do YAML oficial, é
**`NotifyMoiAttributeValueChanges`**. As quatro linhas da tabela mostram a inconsistência
lado a lado:

- `NotifyMOICreation` — "MOI" em maiúsculas (nem a forma do callback nem a do schema)
- `NotifyMOIDeletion` — idem
- `NotifyAttributeValueChanges` — **sem "Moi" nenhum** (o erro em questão)
- `NotifyMoiChanges` — por acaso, a forma canônica do schema

**Alcance:** a forma sem "Moi" aparece **1 vez** contra ~22 das formas corretas, e o
lapso **atravessa as três releases** sem correção (i00, i20, k00). As specs 28.622 e
28.623 não contêm essa grafia nenhuma vez.

**Por que dava erro:** a coluna "Data type" de uma tabela não nomeia nem o callback nem o
schema — só os *headings* ("Notification \<name\>", "Type \<Name\>") mantêm a forma
correta nas três releases. O Extractor, ao tirar o nome da coluna, produzia a grafia
errada e gerava uma regra duplicada (o defeito registrado como 3.1a).

**Conserto (não na spec — é erro da 3GPP):** instruir os agentes a nunca tirarem o nome
de uma coluna de tabela. Ficou no bullet "NAMES: THREE KINDS, THREE PLACES, NEVER
CONVERTED INTO EACH OTHER" do Extractor e na regra correspondente do Validator. O
requisito final é apenas **não duplicar** — sair com o nome errado é tolerável, porque a
fonte é a que erra. Detalhe completo e as contagens por release estão em
`Problems_ProvMnS_LEGACY_ate_20260920.txt` (§3.7).
