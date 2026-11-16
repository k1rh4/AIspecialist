
# RECSYS_NCF 가 평가 관련된 예제 코드임. 
### NCF는 빈칸이 없는데... 평가 Metric의 이해 파트가 있음. 



![[Pasted image 20261116185715.png]]
```python

# 사용자별로 실제 좋아한 아이템 목록을 가져와 반복
for user, pos_items in user_pos_items.items():
    # 현재 사용자의 임베딩 벡터를 가져옴
    user_emb = user_features[user]
    # 모든 아이템 임베딩과 사용자 임베딩을 내적하여
    # 각 아이템의 추천 점수(scores)를 계산
    scores = torch.matmul(item_features, user_emb)
    # 추천 점수가 가장 높은 상위 k개 아이템의 인덱스를 선택
    # topk_indices에는 추천된 아이템의 인덱스가 저장됨
    topk_scores, topk_indices = torch.topk(scores, k=k)
    # GPU 텐서일 수 있으므로 CPU로 이동하고,
    # NumPy 배열을 거쳐 파이썬 리스트로 변환
    topk_indices = topk_indices.cpu().numpy().tolist()
    # 추천한 Top K 중 실제 좋아한 아이템 개수
    hits = 0
    # DCG: 추천 순위를 고려한 실제 추천 품질 점수
    dcg = 0.0
    # IDCG: 이상적으로 추천했을 때 얻을 수 있는 최대 DCG
    idcg = 0.0
    # 사용자가 실제 좋아한 전체 아이템 개수
    # Recall@K 계산의 분모로 사용
    n_pos = len(pos_items)
    # 모델이 추천한 상위 k개 아이템을 순서대로 확인
    # rank: 추천 순위 (0부터 시작)
    # item_idx: 해당 순위에 추천된 아이템 인덱스
    for rank, item_idx in enumerate(topk_indices):
        # 추천한 아이템이 실제 좋아한 아이템에 포함되는지 검사
        if item_idx in pos_items:
            # 정답을 맞힌 추천 아이템 수를 1 증가
            hits += 1
            # 정답 아이템의 순위에 따른 점수를 DCG에 더함
            # 상위 순위일수록 높은 점수를 받음
            # rank가 0부터 시작하므로 rank + 2 사용
            dcg += 1.0 / math.log2(rank + 2)
    # 이상적인 추천 순위에서의 최대 DCG 계산
    # 실제 좋아한 아이템이 k개보다 적으면 그 개수까지만 계산
    for rank in range(min(n_pos, k)):
        # 모든 정답 아이템이 가장 앞 순위부터 배치된다고 가정
        # 각 순위의 점수를 누적하여 IDCG를 계산
        idcg += 1.0 / math.log2(rank + 2)
    # Recall@K
    # 사용자가 실제 좋아한 전체 아이템 중
    # 모델이 Top K 안에서 맞힌 아이템 비율
    recall_u = hits / n_pos
    # Precision@K
    # 모델이 추천한 k개 아이템 중
    # 실제 좋아한 아이템의 비율
    precision_u = hits / k
    # NDCG@K
    # 실제 추천 순위의 품질(DCG)을
    # 이상적인 추천 순위의 품질(IDCG)로 나누어 정규화
    # IDCG가 0이면 0으로 처리하여 0으로 나누는 오류 방지
    ndcg_u = dcg / idcg if idcg > 0 else 0.0
    
    
```



![[Pasted image 20261111160646.png|699]]
users : 유저에 대한 정보 
movies: 아이템 정보 

**==🔴RMSE는 MSE에 제곱근(`sqrt`)을 씌운 것이야.==**
리스트 형태를 받는다고 생각하면 될듯.
![[Pasted image 20261111161119.png|682]]

![[Pasted image 20261116185004.png]]

- **Precision@10:** 추천한 10개 중 실제로 좋아한 게 몇 개야?
	- 추천한 K 중에 사용자가 모델이 예측한 갯수는?
- **Recall@10:** 사용자가 실제로 좋아하는 전체 아이템 중 몇 개를 추천했어?
	- 사용자 선호 영화중에 모델이 예측한 갯수는?

![[Pasted image 20261111161131.png]]

- test batch를 받은 다음 가공을 해야한다. 
- 특정 유저가 아이템별로 상호 작용을 했을수 있으니깐 test_batch 로부터 새로운 dictionary를 만든다.
- dictionary 의 key는 user 가 된다.
- user 가 key인 상호 작용한 item에 대한 rating 값을 저장한다. 
- 유저별로 분리하는과정이라고 보면됨. 
-

# ==🔴이거 시험문제 출제임== 

이 코드에서는 `user_ratings`의 각 원소가 다음 형태라고 볼 수 있어.
(모델의 예측 평점, 실제 평점)

모델이 추천한 것중 사용자가 실제로 좋아한 아이템 / 추천수
모델이 추천한 것중 사용자가 실제로 좋아한 아이템 / 사용자가 좋아한 아이템 전체 
![[Pasted image 20261111161402.png]]
여기서 평가함.
User 별로 Item 에 대한 score 가 쭉 있을때.. 
**==🔴K= 임의 숫자==** 
유저별로 돌면서..  레이팅이 높은 순서로 소팅을 한번 한다. 
recommandated_k 는 이렇게 소팅된 데이터에서 10개만큼 뽑는다. 
==🔴**n_real_relevant_in_top_k**: 모델이 추천한 Top 10 중 사용자가 실제로 좋아한 아이템 수==
**n_real_relevant: 사용자가 실제로 좋아한 전체 아이템 수** (엄청 많을 수도 있겟지? 적을수도 있고 )
n_real_relevant_in_top_k : 특정 유저가 상호 작용한 아이템에 대한 실제 평점을 가져와서 topk를 뽑는다.  
만들어진 10개의 리스트 안에 3.5평점만 저장한다. 



| Metric      | NCF    | NGCF   | 의미                    |
| ----------- | ------ | ------ | --------------------- |
| RMSE        | 사용     | 사용 안 함 | 실제 평점과 예측 평점 차이       |
| Precision@K | 사용     | 사용     | 추천 K개 중 실제 정답의 비율     |
| Recall@K    | 사용     | 사용     | 전체 정답 중 추천 K개에 들어온 비율 |
| NDCG@K      | 사용 안 함 | 사용     | 정답의 추천 순위까지 고려        |


- NCF는 `threshold = 3.5`로 실제 선호 여부를 판정합니다.
- NGCF는 `pos_items`(테스트의 Positive 아이템)로 적중 여부를 판정합니다.
- NCF 실습의 Precision 분모는 변수 `k`가 아니라 **실제 원본대로 `10`** 입니다.
