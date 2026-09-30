# Plano — regra adiada até a seção que completa a informação

## Problema (delimitado após o commit f02913c)
Uma regra de `$ref` externo precisa do filename da spec (ex.: `TS28623_ComDefs.yaml`
para `ErrorResponse`). A seção que ESCREVE o $ref (562, o POST) roda ANTES da seção
que DECLARA o tipo (564). O `f02913c` cobriu o caso "tipo NA memória sem filename"
(caso 3 → declara gap). Falta o caso "tipo NEM na memória NEM na seção" — 562 chega
com memória vazia para ErrorResponse e cai no caso 2 (local, sem filename). Regressão
vs agosto, onde a 562 derivava o filename do RAG local.

## Decisões tomadas com o usuário
- **Quem completa**: o AGENTE da seção que tem a info (a 564), não passo determinístico.
- **Órfão (nunca resolve)**: emite no fim com `validation_passed=False` + nota.
- **Escopo**: SÓ filename externo, por ora (não generalizar para todo missing_context).
- **Condição de adiar**: adiar SÓ se o filename falta nas DUAS fontes —
  (a) seção atual (conteúdo/RAG local) E (b) memória (other_spec_ref_memory).
  Se estiver em qualquer uma, emite completa agora. Adiar é último recurso.

## As 5 peças

### Peça 1 — canal de estado `pending_rules` (novo)
- `graph/state.py`: `pending_rules: Annotated[dict, _merge_ref_memory]` (ou list com
  reducer que sobrevive à troca de seção — decidir dict keyed por (section_id, rule
  identity) para permitir remoção quando resolvido).
- Cada entrada guarda: a regra que a seção quis emitir (rule_type, openapi_object,
  openapi_field, source_name, rule_text, valor parcial), o tipo externo e a spec cujo
  filename falta, e a section_id de origem.
- Builder NÃO zera este canal (diferente de missing_context).

### Peça 2 — Extractor adia em vez de emitir incompleta
- No ponto onde escreve o $ref: se o tipo é EXTERNO (Fase 1 marcou import) E o
  filename não está na seção (RAG/conteúdo local) E não está na memória →
  registra em pending_rules e NÃO emite a regra.
- Se o filename está em qualquer fonte → emite completa (comportamento normal).
- Se está na memória sem filename (caso 3 do f02913c) → mantém como está (declara gap).
- Instrução no prompt do Extractor: explicar o adiamento ("não emita placeholder;
  registre como pendente; a seção que declara o tipo completará").

### Peça 3 — Reflector/Validator não tratam o adiado como faltante
- prompts reflector + validator: uma regra em pending_rules está DELIBERADAMENTE
  adiada — não reportar como missing, não mandar corrigir, não force-include prematuro.
- Passar a lista de pending (compacta) aos nós de missing (Reflector F2, Validator
  S1/S3) para que não a listem como faltante.

### Peça 4 — Extractor de seção futura completa a regra
- No início do processamento de cada seção, DEPOIS da Fase 2 atualizar a memória:
  para cada pending_rule cujo filename agora existe na other_spec_ref_memory →
  emitir a regra completa (com o filename) e remover do pending.
- O agente da seção "vê" o pending relevante e o materializa. Passa por
  Reflector/Validator normal.
- Instrução no prompt: "há regras pendentes aguardando um dado que esta seção pode
  ter; se você tem o filename do tipo X, complete a regra pendente."

### Peça 5 — Builder resolve órfãos no fim
- Ao terminar o run (última seção): pending_rules não resolvidos → emitir como regra
  no output com validation_passed=False + validation_notes "external filename not
  found for <type> from <spec>". Sinalizado, não some.

## Arquivos tocados
- graph/state.py — canal pending_rules + reducer
- nodes/extractor.py — peças 2 e 4
- prompts/extractor_prompts.py — instruções de adiar (2) e completar (4)
- prompts/reflector_prompts.py + validator_prompts.py — peça 3
- nodes/builder.py — peça 5

## Teste
- PerfMnS no i00 (reproduz o ErrorResponse). Esperado: o $ref do ErrorResponse nas
  respostas 4XX/5XX volta a ser `TS28623_ComDefs.yaml#/...` (como agosto), emitido
  quando a 564 declara o tipo. Comparar com o output bom de agosto (20260818_214526).
- Lembrar: TARGET_YAML está em ProvMnS; trocar p/ PerfMnS no teste e reverter.

## Reducer de pending_rules (decidido: apagar de fato)
- _merge_ref_memory só mescla (right wins), NÃO remove. Como um pendente resolvido
  precisa SAIR da lista (senão é reprocessado por seções seguintes → regra duplicada,
  e o Builder o trata como falso órfão), pending_rules usa um REDUCER PRÓPRIO.
- Reducer `_merge_pending(left, right)`:
    * chave = identidade da regra pendente (ex.: (section_id_origem, rule_type,
      openapi_object, openapi_field)).
    * right com um marcador de remoção (ex.: valor None ou {"__remove__": True})
      → apaga a chave de left.
    * right com valor normal → adiciona/atualiza (como o merge atual).
  Assim a seção que completa a regra emite `{chave: None}` para apagar o pendente,
  e a remoção sobrevive ao merge do LangGraph.

## Riscos / pontos de atenção
- Uma regra adiada não pode ser contada no retry-threshold nem virar missing (peça 3).
- Se o tipo nunca for declarado por nenhuma seção do plano, cai no órfão (peça 5).
