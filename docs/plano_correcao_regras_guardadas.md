# Plano — corrigir o fluxo de regras guardadas e a regressão do PerfMnS

Criado em 25/out/2026, depois do run completo do PerfMnS (20/out) e da sonda de
24/out. Causas medidas em `Problems_ProvMnS.txt` (bloco "REGRESSÃO DO PerfMnS",
R.1–R.6).

## Princípio

O fluxo normal vale sempre que a seção, com a memória, diz onde o tipo de um `$ref`
está definido: a regra sai na hora, na própria seção. Guardar só acontece quando
essa informação falta — e então nada pode se perder: a regra guardada é emitida pela
seção que resolve o tipo, ou sai marcada no fim.

Hoje isso falha de duas formas: guarda-se o que não precisava (a 261 guardou o tipo
que ela mesma define; a 251 provavelmente guardou regras sem `$ref`) e perde-se o
que foi guardado (descarte silencioso, deduplicação, nomes que não casam).

## Como cada etapa é verificada

Toda etapa é medida no ProvMnS E no PerfMnS antes de passar à seguinte, com o log
gravado em `data/outputs/diagnostic/`:

- PerfMnS: `tests/investigation/perfmns/test_held_rules_perfmns.py` (251, 253, 255,
  261, 263 pelo grafo real);
- ProvMnS: `tests/investigation/provmns/test_imported_type_origin.py`.

## Etapa 0 — instrumento

O log não mostra o que o Extractor devolveu, por isso a perda da 251 não pôde ser
provada. Registrar, por seção e por tentativa, o que foi para `rules` e o que foi
para `deferred_rules` (tipo de regra, nome, alvo do `$ref`), e quantas regras foram
guardadas em relação ao total. Só observação, sem mudar comportamento.

## Etapa 1 — nada se perde ao guardar (R.2, a 251)

- Código (`extractor_node`, peça 1): uma regra em `deferred_rules` cujo `$ref` não
  aponta para um tipo ainda não resolvido volta ao fluxo normal (`rules`), com log.
  Hoje ela é descartada sem aviso.
- Prompt: deixar explícito o escopo — só vai para `deferred_rules` a regra cujo `$ref`
  aponta para um tipo listado como guardado. Regra sem `$ref` (operação, código de
  status sem corpo, media type) nunca é guardada.
- Verificação: a 251 emite POST, 204 e o corpo; nenhuma regra some.

## Etapa 1b — alinhar o rulesbank ao RAG reindexado (25/out)

O repositório qdrant_3gpp_rag reindexou o corpus com `chunk_spec_sections`
("sections-v3"): cada trecho leva o caminho de títulos e os metadados `defines`
(tipos que o trecho define), `mentions`, `annex`, `spec_number`, `part/parts`. A
TS 28.623 agora está inteira (a definição do NotificationHeader é o trecho C.4.2a,
cujo título nomeia `TS28623_ComDefs.yaml`). O rulesbank não aproveita nada disso.

Coleções usadas aqui e o que muda em cada uma:
- `3gpp_rel18_all` / `3gpp_rel18_28` (corpus, montado no repo do RAG) — só o uso:
  - definição de tipo importado (`search_imported_type_ref`): filtrar primeiro por
    `defines=<Tipo>` (medido: devolve exatamente a definição, em 1º, sozinha; sem
    filtro ela vinha na 10ª/14ª de 20); se nada vier, a busca atual;
  - filename pelo RAG (degrau 3 da cascata): o trecho com `defines` contendo o tipo,
    `annex=True` e um `.yaml` no título dá o arquivo (medido: 8 dos 12 tipos
    importados, contra 2 hoje); arquivos diferentes = opções para a LLM.
- `openapi_reference` — passa a vir do repo do RAG, que já cria a coleção; o rulesbank
  deixa de indexá-la e só consome (hoje seus trechos não têm título da seção e o
  recorte ignora `####`/`#####`).
- `spec_tmp_<id>` (a própria spec, indexada na hora) — passa a usar
  `chunk_spec_sections` do repo do RAG (mesmo formato do corpus), mantendo os
  metadados nossos usados como filtro (`section_id`, `mns`, `stage`).
- `rules_tmp_<id>` — sem mudança (guarda regras, não documento).

FLAG — bloquear o anexo da spec-alvo nas buscas (padrão: ligada). O anexo da spec
cujo YAML estamos gerando é a resposta pronta; não pode entrar como conhecimento.
Bloqueia SÓ o anexo dela (pelo `spec_number`, qualquer versão), nas duas coleções:
- corpus: excluir trechos com `spec_number=<alvo>` E `annex=true`. O `retrieve` do
  repo do RAG só monta `must`; precisa de uma exclusão (`must_not` com filtro
  aninhado). Os anexos das OUTRAS specs continuam liberados (é de lá que vem a
  definição do NotificationHeader).
- `spec_tmp`: indexar com `include_annexes=False` quando a flag está ligada.
O `annex_text` da cascata do filename não muda: ele é lido só pelo código, para o
filename, e nunca entra em prompt.

Verificação: a 253 do PerfMnS grava o NotificationHeader com os campos (via
`defines`); nenhuma busca devolve trecho do anexo da 28.532; ProvMnS e PerfMnS sem
regressão nas sondas.

## Etapa 2 — a seção que define o tipo o reconhece como local (R.3)

- Prompt da Fase 1 (CHECK 1): um tipo que a PRÓPRIA seção define é local — pelo título
  ("Type X", "Enumeration X") ou por uma linha de uma tabela em que a seção define
  tipos ("simple data types", "data types defined in this specification"). Genérico:
  a forma de definir é a da seção, não um formato de uma spec.
- Verificação: a 261 marca `PerfMetricValue` como local, e as regras guardadas dele
  (`observedPerfMetricValue`, `thresholdValue`, `hysteresis`) saem ali, internas.

## Etapa 3 — a cópia certa sobrevive (R.4)

- Builder (`_dedupe_rules`): para a mesma chave, ficar com a regra validada em vez da
  reprovada; entre validadas, a primeira.
- A correção "mover para `deferred_rules`" não foi seguida nas tentativas da 255.
  Com o instrumento da etapa 0, ver primeiro o que o Extractor devolve na tentativa
  2 e 3 — o bloco de correções começa com "Return ONLY the rules listed under RULES TO
  CORRECT", o que pode levar a devolver tudo em `rules`. Corrigir a causa vista, não
  um palpite.
- Verificação: `observedPerfMetricDirection` sai uma vez, validada.

## Etapa 4 — nomes com grafias diferentes casam (R.5)

- Código: ao casar um tipo guardado com um tipo resolvido, comparar os nomes sem
  hífen, sublinhado, espaço e caixa (`Error-Response` ≡ `ErrorResponse`). O `$ref`
  completado usa o nome da seção que resolveu. Cobre grafias com sinais diferentes;
  NÃO cobre nomes com segmento faltando (o caso `NotifyAttributeValueChanges`).
- Verificação: 4XX/5XX do PerfMnS saem externas (`TS28623_ComDefs.yaml`).

## Etapa 5 — Planner e callback (R.6) — decidir com o usuário antes

- O Planner passou a escolher a 248 (Description) e a 249 (URI) do recurso
  `/notificationSink`, provável efeito do texto do CC.5. Rever esse texto para que
  cláusulas que só descrevem um recurso não sejam escolhidas por serem "de
  notificação".
- Critério CALLBACK vs PATH quando o documento descreve o próprio receptor (o YAML do
  PerfMnS modela o lado do consumidor como path). Precisa de decisão.
- Verificação: ProvMnS mantém os callbacks; PerfMnS não tem nenhum; sem as 3
  tentativas por seção.

## Etapa 6 — corpus da TS 28.623 (R.1) — no repositório qdrant_3gpp_rag (usuário)

- Reindexar a 28.623 tratando títulos com mais de 6 `#`; conferir outras specs com o
  mesmo formato.
- Depois, verificar aqui: a 253 grava `NotificationHeader` com os campos; a 255 acha
  a base (`allOf`) e os 5 campos herdados não saem como próprios.

## Etapa 7 — runs completos

ProvMnS e PerfMnS, comparados com os bancos de agosto e setembro. Critério para o
PerfMnS: pelo menos o nível de agosto (17 regras certas) — `allOf` com
NotificationHeader, `hysteresis`, `PerfMetricValue`/`PerfMetricDirection`
validadas, 4XX/5XX externas, nenhum callback; custo de volta perto de US$ 0,20.

## Depois deste plano

Pendentes já registrados no Problems: N.2 (path da DELETE), N.3 (`array(T)`), N.5
(headers sem rule_type).
