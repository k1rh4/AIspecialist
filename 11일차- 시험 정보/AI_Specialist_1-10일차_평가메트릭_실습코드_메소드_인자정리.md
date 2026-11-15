---
tags: [AI-Specialist, 평가메트릭, 실습코드, 빈칸문제]
aliases: [AI Specialist 평가 메트릭 실습 정리]
---
| 평가 목적     | 대표 메트릭                          | 암기할 핵심              |
| --------- | ------------------------------- | ------------------- |
| 분류 성능     | Accuracy, Precision, Recall, F1 | 맞혔는가?               |
| 회귀·시계열 예측 | MSE, RMSE, MAE, MAPE            | 얼마나 틀렸는가?           |
| 추천 시스템    | Recall@K, Precision@K, NDCG@K   | 상위 K개 추천이 좋은가?      |
| LLM       | Loss, Perplexity                | 다음 토큰 예측이 얼마나 어려운가? |
| On-device | Latency, Model Size, Throughput | 얼마나 빠르고 가벼운가?       |
| RAG       | Accuracy, Faithfulness 등        | 답변이 정확하고 근거에 충실한가?  |
# AI Specialist 1~10일차 평가 메트릭 — 실제 메서드·인자·결과 변수

> [!important] 출처 구분
> **실습 코드**, **실습 빈칸**, **정답 파일**을 구분했다. Jupyter 셀 번호는 **1부터 세는 순서**다. `????` 등의 빈칸은 원본에서 그대로 유지하고, 정답은 별도 표기한다.
> 평가 메트릭과 학습 Loss, 자원 효율 지표(모델 크기·속도)는 다르므로 아래에 구분 표시했다.

## 한눈에 보는 전체 표

| 일차   | 실습 파일                                                     | 함수·위치                             | 결과 변수                                                      | 실제 메서드 호출 및 인자                                                                                             | 측정 대상                       | 출처         |
| ---- | --------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------- | ---------- |
| 1~2  | Chapter_5_Excercise_Pretraining.ipynb.bin                 | calc_loss_loader / evaluate_model | `train_loss`, `val_loss`                                   | `calc_loss_loader(data_loader, model, device, num_batches=None)`                                           | 평균 CE 손실                    | 실습 빈칸      |
| 1~2  | Chapter_6_Excercise_Finetuning_Classification.ipynb.bin   | calc_accuracy_loader              | `train_accuracy`, `val_accuracy`, `test_acc`               | `calc_accuracy_loader(data_loader, model, device, num_batches=None)`                                       | 분류 Accuracy                 | 실습 코드      |
| 1~2  | Chapter_7_Exercise_Follow_Instructions_dpo.ipynb.bin      | compute_loader_metric             | `train_loss`, `val_loss`, `train_chosen`, `train_rejected` | `compute_loader_metric(loader)`                                                                            | DPO Loss/선호 로그 확률 (진단값)     | 실습 코드      |
| 3~4  | 01_ResNet18_CIFAR10.ipynb.bin                             | accuracy_top1 / evaluate          | `total_acc`, `total_loss`                                  | `accuracy_top1(logits, targets)`; `evaluate(model, loader, criterion)`                                     | Top-1 Accuracy / Loss       | 실습 코드      |
| 3~4  | 02_ViT_CIFAR10.ipynb.json                                 | evaluate                          | `avg_loss`, `acc`                                          | `evaluate(model, data_loader, loss_history)`                                                               | Accuracy / 평균 NLL           | 실습 코드      |
| 3~4  | 02_ViT_CIFAR10.ipynb.json                                 | plot_confusion_matrix             | `cm`                                                       | `confusion_matrix(all_labels, all_preds)`                                                                  | Confusion Matrix            | 실습 코드      |
| 3~4  | 04_Unet.ipynb.bin                                         | evaluate                          | `val_loss`, `val_losses`                                   | `evaluate(model, val_loader, criterion, device)`                                                           | Validation Loss             | 실습 코드      |
| 6~7  | 1. [AVD] Basic Quantization Lab_answer.ipynb.json         | evaluate / get_model_size         | `accuracy`, `fp32_model_accuracy`, `fp32_model_size`       | `evaluate(model, dataloader)`; `get_model_size(model, data_width=32)`                                      | 분류 정확도/모델 비트 수              | 정답 파일      |
| 6~7  | 2. [AVD] Advanced Quantization Lab_answer.ipynb.bin       | _evaluate / model_evaluate        | `model_perplexity`, `model_size`                           | `self._evaluate()`; `self.get_model_size(data_width=data_width, group_size=group_size)`                    | Perplexity/모델 크기            | 정답 파일      |
| 6~7  | 3. [AVD] Knowledge Distillation_answer.ipynb.bin          | test                              | `test_accuracy_teacher`, `test_accuracy_student_ce` 등      | `test(model, test_loader)`                                                                                 | Teacher/Student Accuracy    | 정답 파일      |
| 6~7  | 4. [AVD] KV Cache Pruning answer.ipynb.bin                | passkey_accuracy                  | `acc`, `accs`                                              | `passkey_accuracy(score_fn, budget, use_attn)`                                                             | Passkey 검색 성공률              | 정답 파일      |
| 6~7  | 5. [AVD] Speculative Decoding Answer.ipynb.bin            | evaluate                          | `model_perplexity`                                         | `evaluate(model, tokenizer)`                                                                               | Perplexity                  | 정답 파일      |
| 8    | ts_practice.ipynb.bin / ts_solution.ipynb.bin             | test                              | `rmse`, `mape`                                             | `mean_squared_error(y_test, test_predictions)`; `mean_absolute_percentage_error(y_test, test_predictions)` | RMSE/MAPE                   | 실습 빈칸 + 정답 |
| 8    | RecSys_NCF.ipynb.bin                                      | 평가 셀                              | `mse`, `rms`                                               | `mean_squared_error(target_rating_list, model_output_list)`; `np.sqrt(mse)`                                | 추천 평점 RMSE                  | 실습 코드      |
| 8    | RecSys_NCF.ipynb.bin                                      | 평가 셀                              | `precisions`, `recalls`                                    | `n_real_relevant_in_top_k / 10`; `/ n_real_relevant`                                                       | Precision@10/Recall@10      | 실습 코드      |
| 8    | RecSys_GCF_practice.ipynb.bin / RecSys_GCF_sol.ipynb.json | evaluate / test                   | `recall`, `precision`, `ndcg`                              | `evaluate(user_features, item_features, test_edge_index, k)`                                               | Recall@K/Precision@K/NDCG@K | 실습 + 정답    |
| 9~10 | 4_RAG_framework_evaluation_with_MCP.ipynb.bin             | parse_response                    | `answer` (1 또는 -1)                                         | `parse_response(response)`                                                                                 | LLM Judge의 Accuracy 판정 파싱   | 실습 코드      |

## 실습 원본 코드 — 함수 정의와 실제 인자


### LLM — 사전학습: Train/Validation Loss
- 파일: `Chapter_5_Excercise_Pretraining.ipynb` · **5번째 셀**

```python
    # ... 중간 코드 생략 ...


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    데이터 로더 전체(또는 일부)를 돌면서 평균 손실을 계산합니다.
    훈련에는 관여하지 않고, 오직 '평가(Evaluation)' 목적으로만 쓰입니다.
    # ... 중간 코드 생략 ...


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    현재 모델의 성능을 훈련 세트와 검증 세트 각각에 대해 평가합니다.
    """
    # ... 중간 코드 생략 ...
    
    with torch.no_grad(): # [중요] 기울기(Gradient) 계산 끔 -> 메모리 절약 및 속도 향상
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        
    model.train() # 평가가 끝나면 다시 훈련 모드로 복귀해야 함
    return train_loss, val_loss
```

### LLM — 분류 파인튜닝: Accuracy
- 파일: `Chapter_6_Excercise_Finetuning_Classification.ipynb` · **1번째 셀**

```python
    # ... 중간 코드 생략 ...
# -----------------------------------------------------------------------------

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    데이터 로더를 순회하며 모델의 정확도(Accuracy)를 계산
    """
    # ... 중간 코드 생략 ...
                logits = model(input_batch)[:, -1, :]  
            
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples
    # ... 중간 코드 생략 ...

        # 에폭이 끝날 때마다 정확도 계산 및 출력
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
```

### LLM — DPO: 진단 Loss 및 선호 확률
- 파일: `Chapter_7_Exercise_Follow_Instructions_dpo.ipynb` · **1번째 셀**

```python
    # ... 중간 코드 생략 ...
    with torch.no_grad():
        # 로더를 순회하며 평균 Loss 계산하는 내부 함수
        def compute_loader_metric(loader):
            total_loss, total_chosen, total_rejected = 0., 0., 0.
            num_batches = min(eval_iter, len(loader))
            if num_batches == 0: return float("nan"), float("nan"), float("nan")
    # ... 중간 코드 생략 ...
            return total_loss/num_batches, total_chosen/num_batches, total_rejected/num_batches

        train_loss, train_chosen, train_rejected = compute_loader_metric(train_loader)
        val_loss, val_chosen, val_rejected = compute_loader_metric(val_loader)

    policy_model.train() # 다시 학습 모드로 전환
    return {
```

### Vision — ResNet: Top-1 Accuracy
- 파일: `01_ResNet18_CIFAR10.ipynb` · **11번째 셀**

```python
    # ... 중간 코드 생략 ...
cfg = TrainConfig()  # config 인스턴스 생성

def accuracy_top1(logits, targets):
    preds = logits.argmax(dim=1)  # 가장 큰 logit의 클래스 선택, logits = [B, C], preds = [B]
    return (preds == targets).float().mean().item()  # top-1 accuracy, scalar

    # ... 중간 코드 생략 ...
        bs = images.size(0)  # batch size
        total_loss += loss.item() * bs  # batch loss 누적
        total_acc  += accuracy_top1(logits.detach(), labels) * bs  # logits값만 복사, batch acc 누적
        n += bs  # 샘플 수 누적

    dt = time.time() - t0  # epoch 수행 시간
    return total_loss / n, total_acc / n, dt  # 평균 loss/acc/시간

@torch.no_grad()   # Autograd off, 함수 내부에서 기울기 계산 멈춤
def evaluate(model, loader, criterion):
    model.eval()  # eval mode, i.e., Dropout 비활성화, BN Running Stats 고정
    total_loss, total_acc = 0.0, 0.0  # 누적
    n = 0  # 샘플 수
    # ... 중간 코드 생략 ...
        bs = images.size(0)  # batch size
        total_loss += loss.item() * bs  # 누적
        total_acc  += accuracy_top1(logits, labels) * bs  # 누적
        n += bs  # 누적
    return total_loss / n, total_acc / n  # 평균 loss/acc

def plot_history(hist):
    epochs = np.arange(1, len(hist['train_loss'])+1)  # epoch index
```

### Vision — ViT: Accuracy/Validation Loss
- 파일: `02_ViT_CIFAR10.ipynb` · **20번째 셀**

```python
    # ... 중간 코드 생략 ...
def evaluate(model, data_loader, loss_history):
    """test_loader로 1회 평가하고 (loss, acc)를 반환합니다."""
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, target in data_loader:
    # ... 중간 코드 생략 ...
            target = target.to(device)

            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += float(loss.item())
            correct_samples += int(pred.eq(target).sum().item())

    avg_loss = total_loss / total_samples
    acc = correct_samples / total_samples

    loss_history.append(avg_loss)
    print(
    # ... 중간 코드 생략 ...
        f"Accuracy: {correct_samples:5d}/{total_samples:5d} ({acc*100:.2f}%)\n"
    )
    return avg_loss, acc
```

### Vision — ViT: Confusion Matrix
- 파일: `02_ViT_CIFAR10.ipynb` · **27번째 셀**

```python
    # ... 중간 코드 생략 ...
import numpy as np

def plot_confusion_matrix(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    # ... 중간 코드 생략 ...
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
```

### Vision — UNet: Validation Loss
- 파일: `04_Unet.ipynb` · **11번째 셀**

```python
    # ... 중간 코드 생략 ...

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    for imgs, masks in dataloader:
    # ... 중간 코드 생략 ...
for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch}/{num_epochs}]  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")
```

### On-device — 양자화: 모델 정확도와 크기
- 파일: `1. [AVD] Basic Quantization Lab_answer.ipynb` · **24번째 셀**

```python
    # ... 중간 코드 생략 ...
fp32_model_accuracy = evaluate(model, dataloader['test'])
fp32_model_size = get_model_size(model)
print(f"fp32 model의 정확도={fp32_model_accuracy:.2f}%")
print(f"fp32 model의 크기={fp32_model_size/MiB:.2f} MiB")
```

### On-device — Advanced Quantization: Perplexity
- 파일: `2. [AVD] Advanced Quantization Lab_answer.ipynb` · **13번째 셀**

```python
    # ... 중간 코드 생략 ...

    @torch.inference_mode()
    def _evaluate(self):
        block_size = EVAL_BLOCK_SIZE
        max_windows = self.testenc.shape[1] // block_size
        nsamples = min(EVAL_MAX_WINDOWS, max_windows)
    # ... 중간 코드 생략 ...
            )
            nlls.append(loss.float() * (block_size - 1))
        return torch.exp(torch.stack(nlls).sum() / (nsamples * (block_size - 1)))

    def get_model_size(self, data_width=16, group_size=-1):
        if group_size != -1:
    # ... 중간 코드 생략 ...
        empty_device_cache()

    def model_evaluate(self, data_width, group_size):
        model_perplexity = self._evaluate()
        model_size = self.get_model_size(data_width=data_width, group_size=group_size)
        print(f'\nmodel perplexity: {model_perplexity:.2f}')
        print(f'model estimated size: {model_size / 1024 / 1024 / 8:.2f} MiB')
        return model_perplexity
```

### On-device — 지식증류: Teacher/Student Accuracy
- 파일: `3. [AVD] Knowledge Distillation_answer.ipynb` · **16번째 셀**

```python
    # ... 중간 코드 생략 ...
teacher_model = VGGCifar9().cuda()
teacher_model.load_state_dict(state_dict)
test_accuracy_teacher = test(teacher_model, test_loader)
```

### On-device — KV Cache: Passkey Accuracy
- 파일: `4. [AVD] KV Cache Pruning answer.ipynb` · **30번째 셀**

```python
    # ... 중간 코드 생략 ...


def passkey_accuracy(score_fn, budget, use_attn):
    """5개 위치에서 passkey retrieval 성공률을 계산합니다."""
    correct = 0

    # ... 중간 코드 생략 ...
        ids, key = make_passkey_prompt(tokenizer, depth=depth, seed=2)

        tokens, _ = prune_and_decode(
            model,
            ids,
            score_fn=score_fn,
    # ... 중간 코드 생략 ...

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        correct += int(key in text)

    return correct / len(DEPTHS)


results = {}
    # ... 중간 코드 생략 ...

    for budget in BUDGETS:
        acc = passkey_accuracy(score_fn, budget, use_attn)
        accs.append(acc)

        print(
```

### On-device — Pruning: Perplexity
- 파일: `5. [AVD] Speculative Decoding Answer.ipynb` · **6번째 셀**

```python
    # ... 중간 코드 생략 ...
def evaluate(model, tokenizer):
    """
    모델의 perplexity를 계산하는 함수입니다.

    # ... 중간 코드 생략 ...

        # 로짓과 레이블 시프트
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]

        # 손실 계산
    # ... 중간 코드 생략 ...
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    # Perplexity 계산 및 반환
    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
```

### Data — 시계열 RMSE/MAPE 정답
- 파일: `ts_solution.ipynb` · **28번째 셀**

```python
    # ... 중간 코드 생략 ...
def test(model, X_test, y_test):
  model.eval()
  with torch.no_grad():
    X_test = X_test.to(device)                     # (N, seq_len, 1)
    y_hat = model(X_test)                          # (N, seq_len, 1)
    test_predictions = y_hat[:, -1, 0]

  test_predictions = test_predictions.cpu().numpy()
  y_test = y_test.cpu().numpy()

  rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
  mape = mean_absolute_percentage_error(y_test, test_predictions)

  return rmse, mape

rmse, mape = test(model, X_test, y_test)
print(f'RMSE: {rmse:.4f}')
print(f'MAPE: {mape:.4f}')

```

### Data — 시계열 RMSE/MAPE 출제 빈칸
- 파일: `ts_practice.ipynb` · **28번째 셀**

```python
    # ... 중간 코드 생략 ...
def test(model, X_test, y_test):
  model.eval()
  with torch.no_grad():
    X_test = X_test.to(device)
    # ... 중간 코드 생략 ...
  y_test = y_test.cpu().numpy()

  rmse = np.sqrt(mean_squared_error(, )) # TODO
  mape = mean_absolute_percentage_error(, ) # TODO

  return rmse, mape

rmse, mape = test(model, X_test, y_test)
print(f'RMSE: {rmse:.4f}')
print(f'MAPE: {mape:.4f}')

```

### Data — NCF 평점 RMSE
- 파일: `RecSys_NCF.ipynb` · **19번째 셀**

```python
    # ... 중간 코드 생략 ...
from sklearn.metrics import mean_squared_error

model_output_list = []
target_rating_list = []
    # ... 중간 코드 생략 ...
    target_rating_list += target_rating_batch

mse = mean_squared_error(target_rating_list, model_output_list)
rms = np.sqrt(mse)
print(f"rms: {rms}")
```

### Data — NCF Precision/Recall
- 파일: `RecSys_NCF.ipynb` · **22번째 셀**

```python
    # ... 중간 코드 생략 ...

  # recall@K
  k= 10
  threshold = 3.5 # relevant item criterion

  for user_id, user_ratings in user_est_true.items():

    # ... 중간 코드 생략 ...

    # precision@k
    precisions[user_id] = n_real_relevant_in_top_k / 10
    # recall@k
    if n_real_relevant:
      recalls[user_id] = n_real_relevant_in_top_k / n_real_relevant

    else:
      recalls[user_id] = 0
```

### Data — NGCF Precision/Recall/NDCG
- 파일: `RecSys_GCF_sol.ipynb` · **18번째 셀**

```python
    # ... 중간 코드 생략 ...
def evaluate(user_features, item_features, test_edge_index, k):
    user_pos_items = defaultdict(list)
    E_test = test_edge_index.size(1)
    for i in range(E_test):
    # ... 중간 코드 생략 ...
            if item_idx in pos_items:
                hits += 1
                dcg += 1.0 / math.log2(rank + 2)
        for rank in range(min(n_pos, k)):
            idcg += 1.0 / math.log2(rank + 2)

        recall_u = hits / n_pos
        precision_u = hits / k
        ndcg_u = dcg / idcg if idcg > 0 else 0.0

        recalls.append(recall_u)
        precisions.append(precision_u)
    # ... 중간 코드 생략 ...
    precision = np.mean(precisions)
    ndcg = np.mean(ndcgs)
    return recall, precision, ndcg
```

### RAG — Judge Accuracy 판정 파싱
- 파일: `4_RAG_framework_evaluation_with_MCP.ipynb` · **24번째 셀**

```python
    # ... 중간 코드 생략 ...
### YOUR CODE HERE ###

def parse_response(response):
    try:
        response = response.lower()
        model_resp = json.loads(response)
        answer = -1
        if "accuracy" in model_resp and (
            (
              model_resp["accuracy"] is True
    # ... 중간 코드 생략 ...
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1

```

## 함수·인자 순서 암기

| 코드 | 인자 순서 |
|---|---|
| `mean_squared_error(y_true, y_pred)` | 정답 → 예측 |
| `mean_absolute_percentage_error(y_true, y_pred)` | 정답 → 예측 |
| `confusion_matrix(y_true, y_pred)` | 정답 → 예측 |
| `calc_accuracy_loader(data_loader, model, device, num_batches=None)` | 로더 → 모델 → 장치 → 배치 제한 |
| `evaluate(user_features, item_features, test_edge_index, k)` | 유저 임베딩 → 아이템 임베딩 → 테스트 엣지 → K |
| `passkey_accuracy(score_fn, budget, use_attn)` | 점수 함수 → KV 예산 → attention 사용 여부 |

> [!note] 구분
> `nn.MSELoss`·`CrossEntropyLoss`·`F.nll_loss`는 **Loss 함수**다. RMSE·MAPE·Accuracy·Recall@K·NDCG@K는 **평가 지표**다. Perplexity는 언어 모델 예측 성능 지표이며, 모델 크기와 tokens/s는 **효율 지표**다.
> 일부 자료의 코드는 시험용 `????` 또는 빈 줄이 남아 있으므로 그대로 실행되지 않을 수 있다.

## 관련 노트

- [[AI_Specialist_1-10일차_정규화_공식_실습코드_정리]]
