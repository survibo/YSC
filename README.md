# grokking

수열 변환 규칙을 작은 Transformer에 학습시키면서 grokking(외운 뒤 한참 지나서 일반화되는 현상)이 언제 일어나는지 본 코드.

1~n 중 서로 다른 정수 m개로 된 수열을 받아 같은 길이의 수열로 바꾸는 규칙을 학습시킨다.

## 설치

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

## 사용법

```bash
# 데이터 split 만들기
python -m src.data --n 50 --m 5 --train-frac 0.01 --test-count 20000 --seed 0

# 학습
python train.py --data data/n50_m5_tr21188_te20000_o1_s0 --rule sort --model m0

# 여러 조건 한 번에 (--multi는 같은 데이터·모델의 run을 한 프로세스에서 같이 돌림)
python sweep.py --data data/n50_m5_tr21188_te20000_o1_s0 \
  --rules sort l1_sum l1_first --models m0 --seeds 0 1 2 3 4 \
  --multi --jobs 4 -- --steps 100000 --stop-after-gen 2000

# 결과 정리, 학습 곡선
python analyze.py "runs/*" --out results.csv
python plot.py "runs/l1_first_*" --out l1_first.png
```

규칙 목록은 `python -m src.rules --list`, 규칙 하나의 성질(대칭, 의존 범위 등)은 `python -m src.rules <규칙>`으로 볼 수 있다.

규칙은 단계를 `>`로 이어서 직접 적을 수도 있음

```
A:asc
I:add(c=3) > J:mul(c=3)
REF:first(op=add)
```

## 모델

| 이름 | 구조 |
|---|---|
| `m0` | 입력 위치 임베딩 + 출력 칸 m개를 한 번에 예측 |
| `minv` | m0에서 입력 위치 임베딩을 뺀 것 (입력 순서에 불변) |
| `meq` | 입력 i번째 자리에서 y_i를 예측 (등변) |
| `gpt` | decoder-only GPT |

기본은 2층, d=128, head 4, AdamW(lr 1e-3, weight decay 1.0), full batch.

## 결과

`results/`에 실험별 결과가 tgz로 있다. 목록은 [`results/README.md`](results/README.md).
