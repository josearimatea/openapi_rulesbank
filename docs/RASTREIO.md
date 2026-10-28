# Documento de Rastreio — OpenAPI Rules Bank

Este documento é um caderno de bordo do projeto. A ideia é registrar, em texto
corrido, as ideias que motivaram cada etapa, o que foi de fato implementado, o
que foi mudado e por quê, e o que foi testado — de forma que sirva como matéria-prima
para a escrita da dissertação de mestrado mais adiante. Não pretende ser documentação
técnica (isso está no README e no código); pretende contar a *história* das decisões.

Ele não é versionado (está no `.gitignore`), justamente por ser um registro pessoal e
em evolução. A cada mudança relevante no projeto, acrescente um parágrafo novo em vez
de reescrever o que já está aqui.

Última atualização: 2026-10-28 (adendo 21–28/out ao fim).

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

---

## Endereçamento e coerência do contrato (18–20/out/2026)

Uma série de consertos com um fio comum: o contrato de cada rule_type — a forma que
`openapi_object`, `openapi_field` e `openapi_value` devem ter — estava escrito em quatro
lugares que não concordavam entre si (o prompt do Extractor, a validação determinística
`rules_check`, as descrições do schema `rules.py` que o structured output leva ao LLM, e
o `rule_types.py` do gerador). O critério de cada decisão foi sempre o YAML oficial e a
referência OpenAPI 3.0.1, nunca uma convenção entre os dois repositórios.

**Parâmetro de query na operação, de path em qualquer nível (§3.22, commit 0cd5be3).**
Os query parameters saíam endereçados no path (`paths./{className}={id}`), o que em
OpenAPI os oferece a todos os métodos. Nos seis YAMLs oficiais o query parameter está
sempre dentro de uma operação (inclusive sob DELETE, no StreamingDataMnS), então o
endereço passou a carregar o método (`paths.<path>.<method>`), exigido pelo prompt e pelo
`rules_check` — qualquer método, não só GET. Já o path parameter aparece nos dois níveis
(no path item quando compartilhado por todos os métodos, como no ProvMnS; dentro da
operação no FaultMnS e no StreamingDataMnS), e o prompt deixou de afirmar que ele é sempre
do path. O "caso b" do §3.22 (path_operation com o método no field) não era defeito: é o
contrato que o gerador lê.

**Endurecimento das checagens (commit 64a520b).** O método no endereço do query parameter
passou a exigir a forma canônica com ponto (`.get`, não `/get`). No callback, o objeto
passou a exigir o método da OPERAÇÃO QUE REGISTRA antes de `.callbacks.` — que não é o
método de entrega: é `put` no ProvMnS e `post` no FaultMnS e no FileDataReportingMnS,
enquanto a entrega é sempre `post` (no field). Medido no banco i20, o Extractor já
produzia a forma certa; a checagem é rede de segurança, não correção de erro ativo.

**Seis defeitos de coerência, achados por leitura cruzada (commit 59c7cfe).** Não vinham
de medição do banco, por isso nunca tinham entrado no Problems. Ficou a regra de processo:
mudar o contrato de um rule_type em qualquer dos quatro lugares obriga rever os outros
três no mesmo passo. Os seis foram consertados:
- **CC.1 (b55e0bb)** — os exemplos do `OpenAPIMapping` (`'NrCellDu'`, `'type'`, `'$ref'`)
  eram formas que o próprio `rules_check` rejeita, e chegavam ao modelo junto do prompt.
  Viraram as formas válidas por rule_type, com placeholders genéricos, sem valores de uma
  spec específica.
- **CC.2 (0dea7c7)** — o `rules_check` não olhava o objeto de `response` e `request_body`;
  um objeto sem método passava e o gerador descartava a regra em silêncio. Passou a exigir
  `….<method>.responses` / `….<method>.requestBody`, aceitando também o caminho de
  callback. Zero falsos positivos nas 41 regras reais do banco.
- **CC.6 (d60ed59)** — propriedade que é array: o Extractor usava quatro formas diferentes.
  Pela referência OpenAPI uma propriedade É um Schema Object e `items` é campo dele, logo
  a forma é `properties.<nome>.items` (7 casos nos YAMLs); o que distingue um elemento de
  outro é o valor (`$ref` para tipo nomeado, tipo simples senão). Sufixos como
  `.description` passaram a ser rejeitados.
- **CC.5 (c49030d)** — o Planner falava em "sete tipos" e omitia o callback, podendo
  excluir seções de notificação. Passou a listar os oito.
- **CC.3 + CC.4 (b0ac2e5)** — a identidade de duplicata era contraditória: o Validator
  ignorava o valor e o Builder sempre o incluía. Passou a ser por tipo: em `request_body` e
  `response` o valor conta (o PATCH do ProvMnS tem 4 media types e o GET 200 tem 3 — eram
  descartados como duplicata, §3.19); nos demais o endereço é único por construção e o
  valor não conta (funde o enum `[UP]`/`[DOWN]`, §3.27, e o callback repetido, §3.24).

**Reconhecer o tipo importado que só uma seção posterior declara (commit 6d398a4).** Ver
a seção seguinte: o CHECK 4 da Fase 1 marca como import pendente um tipo que aparece
sozinho numa célula de tabela de resposta/corpo.

**Do lado do gerador**, o usuário alinhou o `rule_types.py` aos parâmetros (query na
operação; path em qualquer nível). Ficaram anotados, ainda abertos: o endereço do callback
aceita "path registrador com o método no field" e o planner usa o método de entrega como
registrador; o `address` do path_parameter contradiz a própria orientação; o regex aceita
`/get` (tolerância, benigna).

**Infraestrutura (6ac2a99):** o modelo de embedding e o acesso ao Qdrant passaram a vir do
pacote `qdrant_3gpp_rag`.

**Lição.** Vários desses defeitos passaram porque cada conserto foi feito no ponto pedido,
sem reler os outros três lugares do contrato — e alguns foram introduzidos no próprio
passo de conserto (um comentário que contradizia o prompt recém-alterado). Daí a regra de
processo acima.

## De onde vem o nome do arquivo YAML de um `$ref` externo (§3.23/§3.26)

Investigação de 19/out/2026, sobre o maior defeito aberto: tipos importados de outra
spec (ErrorResponse, Dn, Filter, …) saíam com `$ref` **interno** em vez de externo
(`TS28623_ComDefs.yaml#/…`). Perseguindo *por que*, cheguei a onde a informação do
nome do arquivo realmente mora — e a conclusão reorienta o conserto.

**O reconhecimento do import melhorou, mas o filename continua faltando.** Rodando o
teste `tests/investigation/provmns/test_imported_type_origin.py` (ProvMnS, extractor
real, seções em ordem de documento), os 11 tipos importados passaram a ser reconhecidos
com a spec (`spec=TS 28.623` em todos), mas **todos** com `file=—` → 0 `$ref` externos,
10 internos. A spec chega; o nome do arquivo não.

**Por quê: o nome do arquivo não está no texto normativo de nenhuma spec — só nos
anexos OpenAPI.** Verificado:
- Na TS 28.532, `TS28623_ComDefs.yaml` aparece **só no Anexo A** (o YAML-alvo, proibido
  de usar — é a própria resposta). O corpo e a tabela de imports (12.1.1.4.1-2) dão o
  nome do tipo + "TS 28.623", **nunca** o `.yaml`.
- Na TS 28.623 (a spec de origem), `ComDefs` aparece **só no Anexo C** (61×); as 2
  ocorrências "no corpo" são entradas do índice que apontam para o Anexo C. O corpo
  normativo também não traz o nome do arquivo.
- Reordenar as seções **não resolveria** (corrige uma hipótese que eu havia levantado e
  estava errada): a informação não está em nenhuma seção do corpo, em ordem nenhuma.

**Existe, porém, uma CONVENÇÃO normativa de nomes de arquivo — e está no corpus RAG,
fora de anexo.** Recuperada do `3gpp_rel18_all` (cláusula "Conventions for Open API
specification files", 5.2.9.6 / 5.3.6, da família de specs de regras 3GPP):
> o nome = `TSxxyyy` + `_` + (nome da API **ou**, para dados comuns, o token do grupo
> comum — ex. `CommonData`) + `.yaml`; *"unless a specific file name is indicated in the
> Annex of a 3GPP specification"*.
Exemplos normativos citados: `TS29122_CommonData.yaml#/…`, `TSabcde_Nxxx_Yyy.yaml#/…`.

**A ressalva que define a decisão:** a convenção dá a FORMA e o PREFIXO para qualquer
spec (`TS28623_…`), mas o TOKEN exato do grupo comum **varia por família** — a série 29
usa `CommonData`, a TS 28.623 usa **`ComDefs`**. E o que diz que a 28.623 usa justamente
`ComDefs` (e não `CommonData`) só aparece **no Anexo C da 28.623** — ou seja, a própria
exceção que a cláusula prevê ("unless indicated in the Annex"). Então a convenção pura
produziria `TS28623_CommonData.yaml`, que está errado; o certo (`ComDefs`) só se confirma
no anexo.

**Decisão (19/out/2026): usar `ComDefs` para os tipos comuns da TS 28.623, registrando a
ressalva.** O token `ComDefs` vem do conhecimento de que a 28.623 o usa — informação que,
no fim, só o anexo confirma. Portanto NÃO é derivação pura da convenção nem extração do
corpo: é um token fornecido/conhecido, e deve ser tratado como tal (o canal honesto é o
`helper_context`, que por precedência vence a convenção e deixa registrado que veio de
fora da spec). O que É derivável e genérico (e pode sair da convenção 5.2.9.6 do corpus):
o prefixo `TSxxyyy_` e a estrutura `…_<grupo>.yaml#/components/schemas/<Tipo>`. O que NÃO
é derivável sem o anexo: o token exato `ComDefs` da família 28.x.

Mapeamento tipo→arquivo verificado nos 6 YAMLs oficiais (como evidência, não como fonte
da solução): os tipos comuns — ErrorResponse, NotificationHeader, Dn, Filter, Uri,
NotificationId, DateTime, Float, AttributeNameValuePairSet, AttributeValueChangeSet — são
**sempre** `ComDefs` (96 de 108 refs externas, 1-para-1, sem exceção). O tipo genérico
`resources` espalha-se por 8 arquivos NRM diferentes e **não** é derivável do nome —
esse é o §3.10 (anyOf dos NRMs), já delegado ao gerador/helper_context, fora deste escopo.

**Estado:** o CHECK 4 (commit 6d398a4) fechou o reconhecimento do import (spec presente,
tipo marcado como externo/pendente, deferred dispara). Falta ligar o FILENAME: para os
comuns da 28.623, `ComDefs` via helper_context/convenção, com registro no Reflector de
que foi derivado e de onde; para os NRMs, declarar gap sem inventar. Fonte de verdade e
numeração em `Problems_ProvMnS.txt` §3.23/§3.26.

### A solução implementada: cascata de resolução do filename

Decidido e implementado em 19/out/2026, depois de estabelecer que o nome do arquivo não
existe em texto normativo (só em anexo) e que **não** é derivável do número da spec.

**Onde a definição realmente mora (recapitulando, porque orienta a solução):**
- O nome do arquivo (`TS28623_ComDefs.yaml`, `TS28623_GenericNrm.yaml`, …) só aparece
  **nos anexos OpenAPI**: Anexo A da TS 28.532 (o YAML-alvo) e Anexo C da TS 28.623 (as
  definições comuns). O corpo normativo de ambas dá o tipo + "TS 28.623", nunca o `.yaml`.
- Existe uma **convenção de nomes normativa** no corpus, fora de anexo — cláusula
  "Conventions for Open API specification files" (5.2.9.6 / 5.3.6, das specs de regras
  da **série 29**, ex. TS 29.501): o nome = `TSxxyyy` + `_` + (nome da API **ou**, para
  dados comuns, um token como `CommonData`) + `.yaml`. Ela dá a FORMA e o prefixo para
  qualquer spec, mas o TOKEN do grupo comum varia por família (série 29 → `CommonData`;
  série 28 → `ComDefs`), e esse token exato só o anexo confirma. Por isso a convenção
  sozinha não basta — serve de estrutura, não de resposta.

**A ideia central (o que o usuário pediu):** não hardcodar `ComDefs` (se fosse isso,
bastava o prompt). O anexo contém o CATÁLOGO COMPLETO de pares `tipo → arquivo` que o
documento usa — todos os ~10 arquivos da série 28 mais os de outras. A solução extrai
esse catálogo e usa o arquivo CERTO para cada tipo, genérico para qualquer spec/arquivo.

**Determinístico onde é seguro, LLM onde é ambíguo** (verificado contra o anexo do
ProvMnS: 21/21 refs capturados pelo regex, 0 perdidos, nenhuma forma alternativa):
- O código varre o anexo e monta `tipo → {arquivos}`. Dos 13 tipos, 12 mapeiam para UM
  arquivo (ErrorResponse, Dn, Filter, NotificationHeader, Uri, … → `ComDefs`;
  `Subscription` → `FaultMnS`; `TraceJob` → `GenericNrm`). 1 é AMBÍGUO: `resources`
  aparece em 9 arquivos NRM (cada NRM publica o seu).
- **Um arquivo só** → o código resolve (determinístico, seguro).
- **Vários arquivos** → o código NÃO escolhe; entrega as opções e a **LLM** decide pelo
  contexto da cláusula (qual NRM). O `resources` é exatamente o §3.10, aqui tratado pela
  mesma máquina em vez de delegado por fora.

**A cascata de prioridade** (`resolve_external_filename` em `tools/rag_tools.py`), para
no primeiro degrau que oferece algum arquivo:
1. **helper_context** — o arquivo que o USUÁRIO fornecer (upload opcional); maior
   autoridade, override humano explícito. Vazio quando o usuário não dá nada.
2. **anexo do documento** — lido SÓ pela função (regex), nunca enviado a prompt nenhum;
   pulado quando o documento não tem anexo (releases novas antes do YAML existir).
3. **RAG do corpus** — primeiro filtrado pela spec de origem, depois busca ampla.
4. nada responde → filename vazio, regra ADIADA (gap declarado), nunca inventada.
Em cada degrau o código lê o CONJUNTO de arquivos do tipo: um → resolve; vários →
devolve as opções com sufixo `:ambiguous` para a LLM escolher.

**Peças de código:**
- `utils/parsers.py` — a entrada excluída do anexo passa a guardar seu `content` (só o
  anexo), para o reader poder consultá-lo; os excluídos não chegam ao Planner, então a
  regra "não extrair regras do anexo" continua valendo.
- `nodes/reader.py` — monta `annex_text` separado (campo próprio no state), usado apenas
  pela cascata.
- `tools/rag_tools.py` — `_files_for_type_in_text` (extração determinística, tolera os
  escapes `\_`/`\#` do pandoc — sem isso a leitura do anexo falhava em silêncio) e
  `resolve_external_filename` (a cascata, devolve `(filename, options, source)`).
- `nodes/extractor.py` — na gravação da memória: filename único → grava + registra a
  fonte no `note`; ambíguo → grava vazio e registra as opções no `note`.
- `prompts/section_analyzer_prompts.py` — Q2b: a LLM, ao ver "filename AMBIGUOUS … choose
  among: …" no note, escolhe o arquivo cujo NRM/domínio bate com a cláusula; só das
  opções listadas; se a cláusula não decide, deixa vazio e reporta gap.

**Registro de proveniência:** toda resolução vai para o `note` da memória — "resolved via
helper_context/annex/rag" ou "AMBIGUOUS via … among: …" — então o Reflector e o banco
mostram DE ONDE veio cada filename. Nenhum nome é derivado do número da spec; nenhum é
chutado.

### Adendo — 20/out/2026: o adiamento, revisto

Ao revisar o mecanismo de "adiar" (setembro, docs/plano_regra_adiada.md), três
descobertas mudaram a leitura do que foi feito:

**A premissa estava errada quanto ao que vem da seção futura.** Supunha-se que a seção
que declara o tipo traria o nome do arquivo YAML. Não traz: a tabela "Data types
imported" dá `ErrorResponse   TS 28.623 [44]` — o tipo e a SPEC, nunca o `.yaml` (0
ocorrências de `28623_ComDefs` no corpo do i00, 96 no anexo). E a seção que usa o tipo
(`| ErrorResponse | 4xx/5xx | … |`) não tem nem a spec. Então adiar continua tendo
motivo, mas o que se espera da seção futura é a spec, não o arquivo.

**Só a metade de guardar foi implementada.** A seção que usa o tipo não emite a regra e
o Reflector/Validator não reclamam da falta; mas nada a emite depois. Conferido no
código: a lista de adiados só é lida para a própria seção, só o nome do tipo é guardado
(não a regra), e o prompt do Extractor ainda manda não emitir regra de outra seção. Os
prompts afirmavam que "a seção declarante emite a regra" sem que isso existisse. Na
prática, adiar era perder a regra em silêncio — e não faz sentido adiar sem emitir
depois.

**A hipótese do "[44]" (parágrafo de setembro) foi descartada.** A causa medida em 19/out
(Problems §3.23, hipótese b) é o tipo aparecer ISOLADO na célula da tabela de resposta,
longe da cláusula que o pareia com a spec — não o número de referência colado ao
"TS 28.623".

**Medição** (12 tipos importados declarados no corpo da TS 28.532, sem LLM): o anexo
resolve 10; o RAG resolve 2, com ou sem a spec (saber a spec não ajuda o RAG); os 2 que o
anexo não tem não aparecem em nenhum YAML oficial — não são usados.

**Decisões.** Mantém-se o adiamento, completando a parte que falta (a seção futura emite
a regra; no fim, as que sobrarem saem marcadas). A ordem para o nome do arquivo fica:
helper_context > anexo > RAG > fallback final pela convenção `TS<spec>_ComDefs.yaml`. No
fallback, a regra sai com o score do Reflector no limite de aceitação e o Reflector
explica que a cascata não achou e que se usou a convenção. Com anexo, não há o que
adiar. Também ficaram anotados três defeitos da cascata a corrigir: a Fase 2 decide
adiar antes de a cascata resolver o arquivo; as opções ambíguas não chegam à LLM porque
a memória formatada omite a nota; e a busca da definição do tipo é pulada sem motivo
quando a spec ainda é desconhecida.

---

## Adendo — 21 a 28/out/2026: regras guardadas, RAG reindexado, definições únicas

**A regressão do PerfMnS e a lição de medir nos dois.** O run completo do PerfMnS de
20/out saiu pior que os de agosto e setembro (21 regras, 14 validadas, US$ 0,52). Todas as
mudanças de 18–20/out tinham sido medidas só no ProvMnS. A partir daí ficou como regra:
cada mudança é medida no ProvMnS E no PerfMnS, com os logs guardados em
data/outputs/diagnostic/ (o log de 20/out se perdeu em /tmp). A sonda
tests/investigation/perfmns/test_held_rules_perfmns.py passou a ser o instrumento,
rodando pelo grafo real (Extractor → Reflector → Validator → Builder) só as seções
envolvidas. As causas medidas viraram R.1–R.7 em Problems_ProvMnS.txt e o plano em
docs/plano_correcao_regras_guardadas.md.

**O RAG reindexado (25–26/out).** A herança perdida (R.1) não era do fluxo: o corpus
tinha só 7 trechos da TS 28.623, e a definição do NotificationHeader não vinha. O
corpus foi reindexado no repositório qdrant_3gpp_rag (bge-base, trechos por árvore de
títulos, com metadados como `defines`, `annex`, `spec_number`). Aqui o fluxo foi alinhado:
a spec processada sai inteira das buscas no corpus (vem só da coleção dela, indexada no
mesmo formato e sem o anexo — o anexo é o YAML que se quer gerar, não conhecimento); a
definição de um tipo importado é pedida pelo metadado `defines`; uma spec citada como
contexto é buscada só nela; e o Reflector exclui a própria seção pelo metadado de
cláusula, não pelo texto. Com isso o allOf do NotificationHeader voltou.

**As regras guardadas, emitidas uma vez (27/out).** As cópias do hysteresis não vinham
de seções diferentes, e sim do retry da seção que resolve o tipo: a regra guardada era
reemitida a cada tentativa, e — o ponto principal — continuava com o id da seção que a
guardou. O Reflector e o Validator a julgavam pelo veredito daquela seção, onde o tipo
ainda estava guardado, e mandavam "mover para deferred_rules"; o Extractor obedecia e
criava mais cópias. Agora a regra completada é emitida uma única vez, como regra da seção
que resolveu o tipo; o Extractor é avisado, em toda tentativa, das regras que o código já
emitiu e não deve escrever; e uma cópia que ele escreva mesmo assim é descartada pelo
código, que é quem sabe qual é a sua. Junto: regra guardada sem $ref (ou com o tipo já
resolvido) volta ao fluxo em vez de sumir (R.2); `Error-Response` e `ErrorResponse` são o
mesmo tipo (R.5); e uma correção pedida que o Extractor não responde fica pendente até
ser respondida, uma entrada por regra — a nova instrução substitui a antiga, e o
Validator recebe a que já falhou para escrever outra mais clara. Na sonda do PerfMnS as
16 regras saíram validadas, sem cópia.

**O callback espúrio e a decisão de uma definição única (27–28/out).** O PerfMnS ganhava
um callback que o YAML oficial não tem. A investigação mostrou que a mesma ideia errada —
"recurso no consumidor é callback" — estava espalhada: no Extractor (um parágrafo de
julho que um conserto de agosto não removeu), no Planner (o texto do CC.5), na Fase 2, e
cada etapa tinha a sua própria redação de callback vs path. A referência OpenAPI resolve
sem ambiguidade: um callback pertence a uma operação-mãe da API e sua chave é uma
expressão avaliada na requisição/resposta dessa operação. O PerfMnS não tem operação
nenhuma ("Mapping of operations" vazia), logo não pode ter callback; o ProvMnS e o
FileData têm a operação que carrega o endereço. Decidiu-se então ter UMA definição de
cada um dos oito tipos de regra (prompts/rule_type_definitions.py), escrita no código,
com duas partes rotuladas — REFERENCE (citação literal da OpenAPI 3.0.1, com o objeto de
origem) e 3GPP CONVENTION (o que é observado na spec e nos YAMLs, com onde foi
verificado) — e entregue igual a todas as etapas que decidem ou julgam tipo de regra. Uma
categoria intermediária, "derivado", foi descartada: na revisão, a maior parte era a
referência reescrita e o resto estava com o rótulo errado. As redações próprias de cada
prompt saíram. Resultado na sonda do PerfMnS: nenhum callback. Na sonda do ProvMnS
apareceu um path_operation em `/{notificationTarget}` (validado), porque a convenção do
placeholder estava só na definição de callback; isso e outras observações ficaram
guardados em data/outputs/diagnostic/provmns_callbacks_20261028/ para análise.

**O Planner e o contrato (28/out).** Duas descobertas sobre o que o Planner escolhe. A
primeira: o nível LOW do prompt pedia literalmente "sections that provide mapping context
or overview", e ele escolhia cláusulas só em prosa (Description, URI) como "apoio" — que
não geram regra e só gastam tentativas. O LOW passou a ser só tabela que traz regras ou
fatos de que outras seções dependem (como a tabela de tipos importados), e cláusula só em
prosa saiu. A segunda: o Planner decide por uma prévia de 500 caracteres, e a cláusula do
POST do PerfMnS começa por uma tabela de query parameters vazia — as tabelas do corpo e das
respostas ficam depois do corte, e ela às vezes ficava de fora. A linha de cada seção
passou a trazer, lidos por código do texto inteiro, os títulos de todas as tabelas (com a
marca "(empty: n/a)") e os sinais de cada tipo de regra — métodos, paths, códigos de
status, media types, tipos de dado —, com o vocabulário tirado das mesmas definições. Os
sinais são delicados: casamento de padrões lê palavra em outro sentido (um JSON Pointer de
exemplo como path, um valor como código de status). Os casos vistos e o estreitamento de
cada um estão anotados no próprio código, e há uma chave para desligar
(settings.PLANNER_SECTION_SIGNALS) se um run mostrar o Planner enganado por um sinal
errado. Por fim, as seções do formato de arquivo XML entravam no plano porque o contrato
apontava como bloco de stage 3 a cláusula-pai do serviço, que contém também outras
soluções; o localizador passou a apontar a cláusula da própria solução RESTful HTTP, e o
resumo do que fica fora de escopo passou a usar a mesma profundidade.

Discutiu-se também mandar ao Planner o texto inteiro das seções: cabe (o bloco RESTful
da ProvMnS tem ~32 mil tokens), mas é grande para uma decisão de seleção. Ficou-se com
informação estrutural, copiada e não interpretada, que custa pouco: além dos títulos das
tabelas, o nome das colunas de cada uma (lido da linha de cabeçalho nos dois formatos de
tabela que a conversão do .docx produz) e o caminho de títulos acima da seção — uma
cláusula que se chama só "POST" ou "URI" passa a dizer de que recurso é. Os metadados do
RAG (`defines`) foram considerados e descartados: não deixariam lento, mas também são
calculados por regra de texto e só ajudam em seções de tipo, que o título já identifica.
Os sinais ficaram ligados, com a chave e o aviso.

Ainda no Planner, ao medir 5 rodadas por MnS, apareceram duas instabilidades, e as duas
eram de formato, não de julgamento. A primeira: duas seções de borda (o overview de
recursos da PerfMnS; a cláusula URI de `{notificationTarget}` da ProvMnS) caíam entre duas
regras que se contradiziam para elas — o LOW mandava incluir "overview table" e a regra de
prosa mandava excluir "contexto/apoio"; e uma frase herdada do CC.5 ("do NOT exclude it
for lacking a path") servia de brecha para o Planner incluir prosa por ser sobre callback.
A regra de prosa passou a ser decidida pelo campo `tables:` da linha (tabela com conteúdo
nunca é excluída como prosa) e a frase saiu. A segunda: com duas listas livres
(escolhidas, excluídas), o modelo às vezes esquecia uma seção — não a escolhia nem a
excluía — e, em 2 de 5 rodadas da PerfMnS, devolveu identificadores que não eram os
`section_id` oferecidos, e o plano inteiro foi descartado. A resposta passou a ter um
schema montado a cada chamada, com um campo obrigatório por seção oferecida
(`s<section_id>`): o modelo não escreve identificador e não pode deixar seção sem
decisão. Resultado: 5 de 5 planos idênticos nas duas MnS, toda seção com decisão e
motivo, nenhuma resposta inválida. Os sinais por padrão ficaram desligados: medidos com e
sem, a diferença foi de uma seção de borda por MnS, dentro da variação, sem ganho que
compensasse o risco de falso positivo.
