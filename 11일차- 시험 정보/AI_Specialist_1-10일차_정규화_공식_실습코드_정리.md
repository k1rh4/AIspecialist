---
tags:
  - AI-Specialist
  - 실습코드
  - 정규화
  - 빈칸문제
aliases:
  - 1~10일차 정규화 공식 모음
---

# AI Specialist 1~10일차: Norm·정규화·스케일 공식

> [!important] 원본과 정답 구분
> 아래의 **[실습 빈칸]**은 배포된 실습 파일 그대로이며, **[정답]**은 정답 파일 또는 정답을 확인할 수 있는 코드에서 가져왔다. `???`, `## fill this part ##`는 시험 대비용 빈칸이다.  
> 이 문서의 대상은 **정규화 관련 공식과 스케일 계산의 대표 10개 위치**이지, 모든 모델의 성능 평가 metric 전체가 아니다. `norm`이라는 명칭을 사용하는 코드라도 역할은 서로 다를 수 있다.

## 한눈에 보기: 날짜 · 파일 · 함수 · 저장 변수 · 수식

| 일차  | 실습 파일                                                | 함수/위치                                   | 결과 변수                            | 정규화·스케일 수식/호출                                               | 의미                                     | 출처            |
| --- | ---------------------------------------------------- | --------------------------------------- | -------------------------------- | ----------------------------------------------------------- | -------------------------------------- | ------------- |
| 1~2 | `Chapter_4_Excercise_GPT.ipynb`                      | `LayerNorm.forward`                     | **`norm_x`**                     | `(x - mean) / torch.sqrt(var + self.eps)`                   | 토큰 특징의 LayerNorm                       | 실습 빈칸 + 정답 보완 |
| 3~4 | `01_ResNet18_CIFAR10.ipynb`, `02_ViT_CIFAR10.ipynb`  | 데이터 전처리 셀                               | **`train_tfms`, `test_tfms`**    | `T.Normalize(CIFAR10_MEAN, CIFAR10_STD)`                    | RGB 채널별 정규화; 변수에는 *전처리 파이프라인* 저장       | 실습 코드         |
| 3~4 | `02_ViT_CIFAR10.ipynb`                               | `PreNorm.__init__`, `forward`           | **`self.norm`**                  | `nn.LayerNorm(dim)` → `self.norm(x)`                        | LayerNorm *모듈* 저장; 결과는 바로 다음 함수로 전달    | 실습 코드         |
| 6~7 | `4. [AVD] KV Cache Pruning answer.ipynb`             | `h2o_norm_score`                        | **`scores`, `count`** → `return` | `scores / count.view(1, 1, K)`                              | Attention 누적 점수의 Query 수 보정            | 정답 파일         |
| 6~7 | `1. [AVD] Basic Quantization Lab_answer.ipynb`       | `fuse_conv_bn`                          | **`factor`**                     | `bn.weight.data / sqrt(bn.running_var.data + bn.eps)`       | BatchNorm 계수를 Conv에 융합                 | 정답 파일         |
| 6~7 | `2. [AVD] Advanced Quantization Lab_answer.ipynb`    | `smooth_ln_fcs`                         | **`scales`**                     | `act_scales.pow(alpha) / weight_scales.pow(1 - alpha)`      | SmoothQuant의 Activation/Weight 스케일 재분배 | 정답 파일         |
| 6~7 | `5. [AVD] Speculative Decoding Answer.ipynb`         | `get_calib_feat` → `add_batch`          | **`activation_norm`**            | `torch.norm(x, p=2, dim=1) ** 2`                            | Activation L2 norm 제곱                  | 정답 파일         |
| 6~7 | `1. [AVD] Basic Quantization Lab_answer.ipynb`       | `get_quantization_scale_and_zero_point` | **`scale`, `zero_point`**        | `(fp_max - fp_min) / (quantized_max - quantized_min)`       | 실수 범위→정수 범위 양자화 파라미터                   | 정답 파일         |
| 8   | `RecSys_GCF_practice.ipynb` / `RecSys_GCF_sol.ipynb` | `NGCFLayer.forward`                     | **`norm`**                       | `1.0 / torch.sqrt(deg[src] * deg[dst])`                     | 엣지 양끝 노드 차수 정규화                        | 실습 빈칸 + 정답 파일 |
| 10  | `2. Task_1.ipynb`                                    | `BaseRetriever.retrieve`                | **`cosine_scores`**              | `np.dot(...) / (np.linalg.norm(...) * np.linalg.norm(...))` | 임베딩 코사인 유사도                            | 실습 코드         |

**혼동 주의:** 정규화 수식 자체를 작성하는 항목과, 정규화 모듈을 생성하거나 양자화 스케일을 계산하는 항목이 섞여 있다. 표의 *결과 변수*는 각 코드에서 실제 사용한 이름이다.

---

## ① GPT LayerNorm — `norm_x` (1~2일차)

- **파일**: `Chapter_4_Excercise_GPT.ipynb`
- **위치**: `class LayerNorm` → `forward(self, x)`
- **저장 변수**: `mean`, `var`, **`norm_x`**

**[실습 빈칸]**
```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - ????) / torch.sqrt(???? + self.eps)
return self.???? * norm_x + self.????
```

**[정답]**
```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - mean) / torch.sqrt(var + self.eps)
return self.scale * norm_x + self.shift
```

- **암기**: `mean`, `var` → `norm_x`; 결과에 `scale`, `shift` 적용.

## ② NGCF 차수 정규화 — `norm` (8일차)

- **실습 원본**: `RecSys_GCF_practice.ipynb`
- **정답 파일**: `RecSys_GCF_sol.ipynb`
- **함수**: `NGCFLayer.forward`
- **저장 변수**: `deg` → **`norm`**

**[실습 빈칸]**
```python
src, dst = edge_index
deg = torch.zeros(node_features.size(0), device=node_features.device)
deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
norm = ## fill this part ##
```

**[정답 및 적용]**
```python
src, dst = edge_index
deg = torch.zeros(node_features.size(0), device=node_features.device)
deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
norm = 1.0 / torch.sqrt(deg[src] * deg[dst])

src_feat = node_features[src]
dst_feat = node_features[dst]
edge_messages_for_src = self.W1(dst_feat) + self.W2(src_feat * dst_feat)
edge_messages_for_src *= norm.unsqueeze(1)
```

- **암기**: `deg[src] * deg[dst]`의 제곱근 **역수**. `norm`은 노드 특징이 아니라 엣지별 가중치.

## ③ RAG 코사인 유사도 — `cosine_scores` (10일차)

- **파일**: `2. Task_1.ipynb`
- **함수**: `BaseRetriever.retrieve`
- **저장 변수**: `all_embeddings`, `query_embedding` → **`cosine_scores`**

```python
all_embeddings = self.embed_text(all_chunks)
query_embedding = self.embed_text(query)[0]
cosine_scores = np.dot(all_embeddings, query_embedding) / (
    np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
)
top_k_indices = (-cosine_scores).argsort()[:topk]
top_k_chunks = np.array(all_chunks)[top_k_indices]
return top_k_chunks
```

- **암기**: 내적을 두 벡터의 L2 노름의 곱으로 나눈다. `cosine_scores`가 최종 유사도 점수 변수.

## ④ Vision 이미지 Normalize — `train_tfms`, `test_tfms` (3~4일차)

- **파일**: `01_ResNet18_CIFAR10.ipynb`, `02_ViT_CIFAR10.ipynb`
- **위치**: 데이터 전처리 셀 (별도 함수가 아니라 `T.Compose` 선언)
- **저장 변수**: **`train_tfms`, `test_tfms`** — 정규화 결과 텐서가 아닌 변환 파이프라인

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_tfms = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
```

- **내부 개념식**: `(x - mean) / std` (코드에서 수식을 직접 작성한 것이 아님).

## ⑤ ViT PreNorm — `self.norm` (3~4일차)

- **파일**: `02_ViT_CIFAR10.ipynb`
- **함수**: `PreNorm.__init__`, `PreNorm.forward`
- **저장 변수**: **`self.norm`** — 정규화 계층 객체

```python
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
```

- **암기**: GPT는 LayerNorm 수식을 직접 완성, ViT는 `nn.LayerNorm(dim)` 호출.

## ⑥ H2O-Norm — `scores`, `count` (6~7일차)

- **파일**: `4. [AVD] KV Cache Pruning answer.ipynb`
- **함수**: `h2o_norm_score`
- **저장 변수**: **`scores`, `count`**. 최종 보정 결과는 별도 저장 없이 `return`.

```python
def h2o_norm_score(attn_kv, layer, budget):
    B, H, Q, K = attn_kv.shape
    recent_size = budget // 2
    scores = attn_kv.sum(dim=-2)
    count = torch.arange(
        K, 0, -1,
        device=attn_kv.device,
        dtype=scores.dtype,
    )
    scores[..., -recent_size:] = float("inf")
    return scores / count.view(1, 1, K)
```

- **암기**: Query 방향 누적 `sum(dim=-2)` → `count`로 나눔.

## ⑦ BatchNorm Fusion — `factor` (6~7일차)

- **파일**: `1. [AVD] Basic Quantization Lab_answer.ipynb`
- **함수**: `fuse_conv_bn`
- **저장 변수**: **`factor`**

```python
def fuse_conv_bn(conv, bn):
    assert conv.bias is None
    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(-bn.running_mean.data * factor + bn.bias.data)
    return conv
```

- **암기**: `factor = BN의 gamma / sqrt(running_var + eps)`.

## ⑧ SmoothQuant — `scales` (6~7일차)

- **파일**: `2. [AVD] Advanced Quantization Lab_answer.ipynb`
- **함수**: `smooth_ln_fcs`
- **저장 변수**: **`weight_scales`**, **`scales`**

```python
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
    )
    scales.clamp(min=1e-5).to(device).to(dtype)
    ln.weight.div_(scales)
    ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
```

- **주의**: 위는 정답 코드에서 핵심 흐름만 발췌한 것이며 전제 검증 `assert` 등을 생략했다. **실습 원본**에서는 공식이 `YOUR CODE STARTS HERE`에 해당할 수 있으니, **정답 코드와 실습 빈칸을 동일시하지 말 것**.

## ⑨ Activation L2 Norm — `activation_norm` (6~7일차)

- **파일**: `5. [AVD] Speculative Decoding Answer.ipynb`
- **함수**: `get_calib_feat` 내부 `add_batch`
- **저장 변수**: **`activation_norm`**

```python
x = x.type(torch.float32)
activation_norm = torch.norm(x, p=2, dim=1) ** 2
input_dict[name] += activation_norm / nsamples_dict[name]
```

- **암기**: `p=2`, `dim=1`, `** 2`를 혼동하지 않기.

## ⑩ Quantization Scale — `scale`, `zero_point` (6~7일차)

- **파일**: `1. [AVD] Basic Quantization Lab_answer.ipynb`
- **함수**: `get_quantization_scale_and_zero_point`
- **저장 변수**: **`scale`**, **`zero_point`**

```python
def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    zero_point = round(quantized_min - fp_min / scale)

    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else:
        zero_point = round(zero_point)
    return scale, int(zero_point)
```

- **암기**: `scale`은 **실수 구간 폭 / 정수 구간 폭**, `zero_point`는 정수 범위의 영점 이동값.

---

## 시험 직전 5줄 요약

1. **GPT**: `norm_x = (x - mean) / sqrt(var + eps)` → `scale * norm_x + shift`
2. **NGCF**: `norm = 1 / sqrt(deg[src] * deg[dst])`
3. **H2O-Norm**: `scores = attn_kv.sum(dim=-2)` → `return scores / count.view(1,1,K)`
4. **BatchNorm Fusion**: `factor = gamma / sqrt(running_var + eps)`
5. **SmoothQuant**: `scales = act_scales.pow(alpha) / weight_scales.pow(1-alpha)`

## 참고: 원본 확인 기준

- GPT: `Chapter_4_Excercise_GPT.ipynb`에는 **`????` 실습 빈칸**이 존재함.
- NGCF: `RecSys_GCF_practice.ipynb`에는 **`norm = ## fill this part ##`**, `RecSys_GCF_sol.ipynb`에 완성 공식이 존재함.
- 6~7일차의 일부 코드는 파일명에 **`answer`**가 붙은 정답 파일에서 확인함.
- 위의 각 코드 조각은 학습 가독성을 위해 해당 수식의 앞뒤 핵심 문맥만 발췌했으므로, 함수 전체를 그대로 실행하는 완전한 프로그램은 아님.
