# Speculative Decoding 실습 빈칸 문제 · 정답

> [!info] 원본 대조
> - 문제: `5. [AVD] Speculative Decoding.ipynb`
> - 정답: `5. [AVD] Speculative Decoding Answer.ipynb`
> - 원본과 정답 파일의 같은 코드 셀을 직접 비교하여 작성
> - 정답은 Obsidian 접기 콜아웃으로 구성 (기본 접힘 · 클릭하여 펼치기)
> - 원본 코드에 있는 `TODO`, `YOUR CODE`와 주변 문맥을 유지
> - **사용 방법:** Obsidian 읽기 보기/라이브 미리보기에서 `정답 보기`를 클릭

## 목차

1. [[#문제 01 — Magnitude Pruning — 가중치 중요도 및 마스크]]
2. [[#문제 02 — Wanda — Activation Norm]]
3. [[#문제 03 — Wanda — 가중치 중요도 및 마스크]]
4. [[#문제 04 — Speculative Decoding — Draft 생성과 Target 검증]]
5. [[#문제 05 — Speculative Decoding — Accept / Reject]]
6. [[#문제 06 — EAGLE — Tree Attention Mask]]
7. [[#문제 07 — EAGLE — Tree Verification]]
8. [[#문제 08 — EAGLE — Tree Acceptance]]
9. [[#문제 09 — DFlash — Parallel Block Drafting]]

---

## 문제 01 — Magnitude Pruning — 가중치 중요도 및 마스크

**원본 노트북 코드 셀:** `9`

### 문제
```python
@torch.no_grad()
def prune_magnitude(model, sparsity):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            ##################### YOUR CODE STARTS HERE #####################
            num_elements = 
            num_zeros =   
            importance =  
            threshold =  
            mask = 
            ##################### YOUR CODE ENDS HERE #######################
            W.mul_(mask)
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> @torch.no_grad()
> def prune_magnitude(model, sparsity):
>     for n, m in model.named_modules():
>         if isinstance(m, nn.Linear) and "lm_head" not in n:
>             W = m.weight.data
>             ##################### YOUR CODE STARTS HERE #####################
>              # 전체 가중치 개수
>             num_elements = W.numel()
>              # 제거할 가중치 개수
>             num_zeros = round(num_elements * sparsity)
>              # 각 가중치의 중요도 = 절대값
>             importance = torch.abs(W)
>              # 제거 대상의 기준값(임계값) 계산 
>             threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]
>              # 임계값보다 큰 가중치만 유자 
>             mask = importance > threshold
>             ##################### YOUR CODE ENDS HERE #######################
>             W.mul_(mask)
> ```

---

## 문제 02 — Wanda — Activation Norm

**원본 노트북 코드 셀:** `15`

### 문제
```python
@torch.no_grad()
def get_calib_feat(model, tokenizer, samples):
    input_dict = dict()
    nsamples_dict = dict()
    def add_batch(m, x, y, name):
        if name not in input_dict:
            input_dict[name] = torch.zeros((m.weight.data.shape[1]), device=m.weight.data.device)
            nsamples_dict[name] = 0

        if isinstance(x, tuple):
            x = x[0]

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        tmp = x.shape[0]
        if len(x.shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
        x = x.t()

        input_dict[name] *= nsamples_dict[name] / (nsamples_dict[name] + tmp)
        nsamples_dict[name] += tmp

        x = x.type(torch.float32)
        ##################### YOUR CODE STARTS HERE #####################
        # activation_norm을 계산하세요.
        # x.shape => (hidden_size, batch_size)
        activation_norm = 엑스피딤제
        # activation_norm.shape => (hidden_size)
        ##################### YOUR CODE ENDS HERE #######################
        input_dict[name] += activation_norm / nsamples_dict[name]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in name:
            hooks.append(
                m.register_forward_hook(
                    partial(add_batch, name=name)))

    print("Collecting norm of input activations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for key in input_dict.keys():
        input_dict[key].sqrt_()

    for hook in hooks:
        hook.remove()
    return input_dict
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> @torch.no_grad()
> def get_calib_feat(model, tokenizer, samples):
>     input_dict = dict()
>     nsamples_dict = dict()
>     def add_batch(m, x, y, name):
>         if name not in input_dict:
>             input_dict[name] = torch.zeros((m.weight.data.shape[1]), device=m.weight.data.device)
>             nsamples_dict[name] = 0
>
>         if isinstance(x, tuple):
>             x = x[0]
>
>         if len(x.shape) == 2:
>             x = x.unsqueeze(0)
>         tmp = x.shape[0]
>         if len(x.shape) == 3:
>             x = x.reshape((-1, x.shape[-1]))
>         x = x.t()
>
>         input_dict[name] *= nsamples_dict[name] / (nsamples_dict[name] + tmp)
>         nsamples_dict[name] += tmp
>
>         x = x.type(torch.float32)
>         ##################### YOUR CODE STARTS HERE #####################
>         # activation_norm을 계산하세요.
>         # x.shape => (hidden_size, batch_size)
>         activation_norm = torch.norm(x, p=2, dim=1) ** 2
>         # activation_norm.shape => (hidden_size)
>         ##################### YOUR CODE ENDS HERE #######################
>         input_dict[name] += activation_norm / nsamples_dict[name]
>
>     hooks = []
>     for name, m in model.named_modules():
>         if isinstance(m, nn.Linear) and "lm_head" not in name:
>             hooks.append(
>                 m.register_forward_hook(
>                     partial(add_batch, name=name)))
>
>     print("Collecting norm of input activations...")
>     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>
>     pbar = tqdm.tqdm(samples)
>     for input_ids in pbar:
>         input_ids = input_ids.to(device)
>         model(input_ids)
>
>     for key in input_dict.keys():
>         input_dict[key].sqrt_()
>
>     for hook in hooks:
>         hook.remove()
>     return input_dict
> ```

---

## 문제 03 — Wanda — 가중치 중요도 및 마스크

**원본 노트북 코드 셀:** `18`

### 문제
```python
@torch.no_grad()
def prune_wanda(model, sparsity, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            ##################### YOUR CODE STARTS HERE #####################
            row, col =
            num_zeros_per_row =  
            importance = 
            threshold = 
            mask =
            ##################### YOUR CODE ENDS HERE #######################
            W.mul_(mask)
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> @torch.no_grad()
> def prune_wanda(model, sparsity, input_feat):
>     for n, m in model.named_modules():
>         if isinstance(m, nn.Linear) and "lm_head" not in n:
>             W = m.weight.data
>             ##################### YOUR CODE STARTS HERE #####################
>             row, col = W.shape
>             num_zeros_per_row = round(col * sparsity)
>             importance = torch.abs(W) * input_feat[n]
>             threshold = torch.kthvalue(importance, num_zeros_per_row, dim=1)[0]
>             mask = importance > threshold.reshape(row, 1)
>             ##################### YOUR CODE ENDS HERE #######################
>             W.mul_(mask)
> ```

---

## 문제 04 — Speculative Decoding — Draft 생성과 Target 검증

**원본 노트북 코드 셀:** `27`

### 문제
```python
# =====================================================================
# 3-2. Draft Generation
# =====================================================================

@torch.inference_mode()
def draft_tokens(model, x, cache, gamma):
  tokens = []

  ######################### 실습 시작 ########################

  # TODO: 다음 draft token을 생성하세요.
  # HINT: greedy_step()을 사용해 draft token을 gamma개 순차적으로 생성하세요.
  for _ in range( ):
    x, cache =
    tokens.append(x)

  ######################### 실습 끝 ########################

  return torch.cat(tokens, dim=-1), cache


# =====================================================================
# 3-3. Target Verification
# =====================================================================

@torch.inference_mode()
def verify_draft(target, x, drafted, cache):

  ######################### 실습 시작 ########################

  # TODO : 마지막 committed token과 draft token들을 하나의 sequence로 연결하세요.
  # HINT: torch.cat()을 사용해 마지막 committed token과 draft token들을 sequence 방향으로 연결하세요.
  verify_ids =

  result = target(
    verify_ids,
    past_key_values=cache,
    use_cache=True
  )

  preds = result.logits[0].argmax(-1)

  ######################### 실습 끝 ########################

  return preds, result.past_key_values
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> # =====================================================================
> # 3-2. Draft Generation
> # =====================================================================
>
> @torch.inference_mode()
> def draft_tokens(model, x, cache, gamma):
>   tokens = []
>
>   ######################### 실습 시작 ########################
>
>   # TODO: 다음 draft token을 생성하세요.
>   # HINT: greedy_step()을 사용해 draft token을 gamma개 순차적으로 생성하세요.
>   for _ in range(gamma):
>     x, cache = greedy_step(model, x, cache)
>     tokens.append(x)
>
>   ######################### 실습 끝 ########################
>
>   return torch.cat(tokens, dim=-1), cache
>
>
> # =====================================================================
> # 3-3. Target Verification
> # =====================================================================
>
> @torch.inference_mode()
> def verify_draft(target, x, drafted, cache):
>
>   ######################### 실습 시작 ########################
>
>   # TODO : 마지막 committed token과 draft token들을 하나의 sequence로 연결하세요.
>   # HINT: torch.cat()을 사용해 마지막 committed token과 draft token들을 sequence 방향으로 연결하세요.
>   verify_ids = torch.cat([x, drafted], dim=-1)
>
>   result = target(
>     verify_ids,
>     past_key_values=cache,
>     use_cache=True
>   )
>
>   preds = result.logits[0].argmax(-1)
>
>   ######################### 실습 끝 ########################
>
>   return preds, result.past_key_values
> ```

---

## 문제 05 — Speculative Decoding — Accept / Reject

**원본 노트북 코드 셀:** `28`

### 문제
```python
# =====================================================================
# 3-4. Accept / Reject
# =====================================================================

def accept_draft(drafted, preds, gamma):

  ######################### 실습 시작 ########################

  #TODO: Draft와 Target prediction을 앞에서부터 비교하고 첫 mismatch에서 멈추세요.
  n = 0
  for i in range(gamma):
    if
      n += 1
    else:
      break

  # accept된 draft token 뒤에 Target의 prediction 하나를 추가
  new_tokens = torch.cat([drafted[:, :n],preds[n].view(1, 1)],dim=1)

  ######################### 실습 끝 ########################

  return new_tokens, n
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> # =====================================================================
> # 3-4. Accept / Reject
> # =====================================================================
>
> def accept_draft(drafted, preds, gamma):
>
>   ######################### 실습 시작 ########################
>
>   #TODO: Draft와 Target prediction을 앞에서부터 비교하고 첫 mismatch에서 멈추세요.
>   n = 0
>   for i in range(gamma):
>     if drafted[0, i] == preds[i]:
>       n += 1
>     else:
>       break
>
>   # HINT: accept된 draft token 뒤에 Target의 prediction 하나를 추가하세요.
>   new_tokens = torch.cat([drafted[:, :n],preds[n].view(1, 1)],dim=1)
>
>   ######################### 실습 끝 ########################
>
>   return new_tokens, n
> ```

이 코드는 **Speculative Decoding(추측 디코딩)에서 Draft 모델이 생성한 토큰을 Target 모델이 검증한 뒤, 몇 개를 받아들일지 결정하는 코드**야.핵심은 **Draft와 Target의 예측을 앞에서부터 비교하다가, 처음으로 다른 토큰이 나오면 비교를 중단하는 것**이야.


---

## 문제 06 — EAGLE — Tree Attention Mask

**원본 노트북 코드 셀:** `39`

### 문제
```python
# =====================================================================
# 4-3. [실습 1] Tree Attention Mask
# =====================================================================

import eagle.model.student_tree as student_tree

def build_tree_mask_and_positions(
  mask_index_list,
  total_tokens,
  device=None
):
  # 자기 자신은 항상 볼 수 있음
  tree_mask = torch.eye(
    total_tokens + 1,
    dtype=torch.bool,
    device=device
  )
  tree_mask[:, 0] = True

  ######################### 실습 시작 ########################

  # ToDo: child가 parent가 볼 수 있는 node들을 그대로 볼 수 있게 하세요.
  # |= 는 OR 결과를 왼쪽에 저장하는 연산입니다.
  for child, parent in enumerate(mask_index_list, start=1):


  ######################### 실습 끝 ########################

  # 보이는 node 수를 이용해 Tree depth 계산
  tree_position_ids = tree_mask.sum(dim=1).long() - 1
  tree_mask = tree_mask.float()[None, None]
  return tree_mask, tree_position_ids

# 구현을 실제 EAGLE에 연결
student_tree.build_tree_mask_and_positions = build_tree_mask_and_positions
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> # =====================================================================
> # 4-3. [실습 1] Tree Attention Mask
> # =====================================================================
>
> import eagle.model.student_tree as student_tree
>
> def build_tree_mask_and_positions(
>   mask_index_list,
>   total_tokens,
>   device=None
> ):
>   # 자기 자신은 항상 볼 수 있음
>   tree_mask = torch.eye(
>     total_tokens + 1,
>     dtype=torch.bool,
>     device=device
>   )
>   tree_mask[:, 0] = True
>
>   ######################### 실습 시작 ########################
>
>   # ToDo: child가 parent가 볼 수 있는 node들을 그대로 볼 수 있게 하세요.
>   # |= 는 OR 결과를 왼쪽에 저장하는 연산입니다.
>   for child, parent in enumerate(mask_index_list, start=1):
>     tree_mask[child] |= tree_mask[parent]
>
>   ######################### 실습 끝 ########################
>
>   # 보이는 node 수를 이용해 Tree depth 계산
>   tree_position_ids = tree_mask.sum(dim=1).long() - 1
>   tree_mask = tree_mask.float()[None, None]
>   return tree_mask, tree_position_ids
>
> # 구현을 실제 EAGLE에 연결
> student_tree.build_tree_mask_and_positions = build_tree_mask_and_positions
> ```

---

## 문제 07 — EAGLE — Tree Verification

**원본 노트북 코드 셀:** `41`

### 문제
```python
# =====================================================================
# 4-4. [실습 2] Tree Verification
# =====================================================================

import eagle.model.ea_model as ea_module

def tree_decoding(
  model,
  tree_candidates,
  past_key_values,
  tree_position_ids,
  input_ids,
  retrieve_indices,
):
  position_ids = tree_position_ids + input_ids.shape[1]
  if position_ids.dim() == 1:
    position_ids = position_ids.unsqueeze(0)


  ######################### 실습 시작 ########################

  # To Do: Tree candidate 전체를 Target model에 한 번에 입력하세요.
  outputs, tree_logits, hidden_state = model(  )

  ######################### 실습 끝 ########################

  # 아래 부분은 EAGLE-3 내부 처리를 위해 제공
  if model.use_eagle3:
    device = model.ea_layer.lm_head.weight.device
    states = [x.to(device) for x in outputs["hidden_states"]]
    hidden_state = torch.cat(states, dim=-1)
  logits = tree_logits[0, retrieve_indices]

  return logits, hidden_state, outputs


# 구현을 EAGLE에 연결
ea_module.tree_decoding = tree_decodings
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> # =====================================================================
> # 4-4. [실습 2] Tree Verification
> # =====================================================================
>
> import eagle.model.ea_model as ea_module
>
> def tree_decoding(
>   model,
>   tree_candidates,
>   past_key_values,
>   tree_position_ids,
>   input_ids,
>   retrieve_indices,
> ):
>   position_ids = tree_position_ids + input_ids.shape[1]
>   if position_ids.dim() == 1:
>     position_ids = position_ids.unsqueeze(0)
>
>
>   ######################### 실습 시작 ########################
>
>   # To Do: Tree candidate 전체를 Target model에 한 번에 입력하세요.
>   outputs, tree_logits, hidden_state = model(
>     tree_candidates,
>     output_orig=True,
>     past_key_values=past_key_values,
>     position_ids=position_ids,
>   )
>
>   ######################### 실습 끝 ########################
>
>   # 아래 부분은 EAGLE-3 내부 처리를 위해 제공
>   if model.use_eagle3:
>     device = model.ea_layer.lm_head.weight.device
>     states = [x.to(device) for x in outputs["hidden_states"]]
>     hidden_state = torch.cat(states, dim=-1)
>   logits = tree_logits[0, retrieve_indices]
>
>   return logits, hidden_state, outputs
>
>
> # 구현을 EAGLE에 연결
> ea_module.tree_decoding = tree_decoding
> ```

---

## 문제 08 — EAGLE — Tree Acceptance

**원본 노트북 코드 셀:** `42`

### 문제
```python
# =====================================================================
# 4-5. [실습 3] Tree Acceptance
# =====================================================================

def evaluate_posterior(
  logits,
  candidates,
  logits_processor,
):

  # 본 실습에서는 Greedy Decoding만 사용
  assert logits_processor is None

  # 각 branch에서 Draft와 Target prediction 비교
  matches = (
    candidates[:, 1:].to(logits.device)
    == logits[:, :-1].argmax(dim=-1)
  ).int()

  # 첫 mismatch 전까지 연속으로 맞은 token 수
  accept_lengths = torch.cumprod(
    matches,
    dim=1
  ).sum(dim=1)


  ######################### 실습 시작 ########################

  # To Do: 가장 길게 일치한 branch와 그 accept length를 선택하세요.
  best_candidate =
  accept_length =

  ######################### 실습 끝 ########################

  sample_p = logits[
    best_candidate,
    accept_length
  ]

  return best_candidate, accept_length, sample_p


# 구현을 EAGLE에 연결
ea_module.evaluate_posterior = evaluate_posterior
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> # =====================================================================
> # 4-5. [실습 3] Tree Acceptance
> # =====================================================================
>
> def evaluate_posterior(
>   logits,
>   candidates,
>   logits_processor,
> ):
>
>   # 본 실습에서는 Greedy Decoding만 사용
>   assert logits_processor is None
>
>   # 각 branch에서 Draft와 Target prediction 비교
>   matches = (
>     candidates[:, 1:].to(logits.device)
>     == logits[:, :-1].argmax(dim=-1)
>   ).int()
>
>   # 첫 mismatch 전까지 연속으로 맞은 token 수
>   accept_lengths = torch.cumprod(
>     matches,
>     dim=1
>   ).sum(dim=1)
>
>
>   ######################### 실습 시작 ########################
>
>   # To Do: 가장 길게 일치한 branch와 그 accept length를 선택하세요.
>   best_candidate = accept_lengths.argmax()
>   accept_length = accept_lengths[best_candidate]
>
>   ######################### 실습 끝 ########################
>
>   sample_p = logits[
>     best_candidate,
>     accept_length
>   ]
>
>   return best_candidate, accept_length, sample_p
>
>
> # 구현을 EAGLE에 연결
> ea_module.evaluate_posterior = evaluate_posterior
> ```

---

## 문제 09 — DFlash — Parallel Block Drafting

**원본 노트북 코드 셀:** `50`

### 문제
```python
# =====================================================================
# 5-1. [실습] Parallel Block Drafting
#   한 번의 DFlash forward로 여러 draft token을 동시에 생성합니다.
# =====================================================================

import dflash.student_block as student_block


def parallel_block_draft(
  model,
  target_hidden,
  noise_embedding,
  draft_position_ids,
  past_key_values_draft,
  output_head,
  verify_size,
):

  global LAST_DRAFT

  # DFlash drafter를 한 번만 forward
  draft_hidden = model(
    target_hidden=target_hidden,
    noise_embedding=noise_embedding,
    position_ids=draft_position_ids,
    past_key_values=past_key_values_draft,
    use_cache=True,
  )

  # 실제로 제안할 draft token 수
  num_draft = verify_size - 1


  ######################### 실습 시작 ########################

  # To Do: 마지막 num_draft개의 hidden state만 선택하세요.
  draft_hidden =

  # To Do: model.compute_logits()을 사용해 모든 위치의 vocabulary logits을 계산하세요.
  draft_logits =

  # To Do: torch.argmax()를 사용해 모든 위치의 token을 동시에 선택하세요.
  draft_tokens =

  ######################### 실습 끝 ########################

  # 첫 번째 draft block만 저장
  if LAST_DRAFT is None:
    LAST_DRAFT = draft_tokens.detach().clone()

  return draft_tokens

  return draft_tokens


# 구현을 DFlash generation에 연결
student_block.parallel_block_draft = parallel_block_draft
```

> [!success]- 정답 보기 (클릭하여 펼치기)
> ```python
> # =====================================================================
> # 5-1. [실습] Parallel Block Drafting
> #   한 번의 DFlash forward로 여러 draft token을 동시에 생성합니다.
> # =====================================================================
>
> import dflash.student_block as student_block
>
>
> def parallel_block_draft(
>   model,
>   target_hidden,
>   noise_embedding,
>   draft_position_ids,
>   past_key_values_draft,
>   output_head,
>   verify_size,
> ):
>
>   global LAST_DRAFT
>
>   # DFlash drafter를 한 번만 forward
>   draft_hidden = model(
>     target_hidden=target_hidden,
>     noise_embedding=noise_embedding,
>     position_ids=draft_position_ids,
>     past_key_values=past_key_values_draft,
>     use_cache=True,
>   )
>
>   # 실제로 제안할 draft token 수
>   num_draft = verify_size - 1
>
>
>   ######################### 실습 시작 ########################
>
>   # To Do: 마지막 num_draft개의 hidden state만 선택하세요.
>   draft_hidden = draft_hidden[:, -num_draft:, :]
>
>   # To Do: model.compute_logits()을 사용해 모든 위치의 vocabulary logits을 계산하세요.
>   draft_logits = model.compute_logits(draft_hidden, output_head)
>
>   # To Do: torch.argmax()를 사용해 모든 위치의 token을 동시에 선택하세요.
>   draft_tokens = torch.argmax(draft_logits, dim=-1)
>
>   ######################### 실습 끝 ########################
>
>   # 첫 번째 draft block만 저장
>   if LAST_DRAFT is None:
>     LAST_DRAFT = draft_tokens.detach().clone()
>
>   return draft_tokens
>
>   return draft_tokens
>
>
> # 구현을 DFlash generation에 연결
> student_block.parallel_block_draft = parallel_block_draft
> ```

---

## 원본과 정답의 차이 관련 메모

- `Tree Verification` 셀은 빈칸 외에 마지막 연결 코드도 `ea_module.tree_decoding = tree_decodings` → `ea_module.tree_decoding = tree_decoding`으로 달라져 있다. 위 문제/정답에는 각각 원본을 그대로 보존했다.
- `Tree Verification`의 `model(...)` 빈칸은 정답에서 여러 줄의 인자로 채워져 있으므로 시험의 '4줄 이하' 조건과 다를 수 있다.
- 이 문서는 실습 코드의 실제 차이만 문제화했으며, 추가 예상문제는 포함하지 않았다.
