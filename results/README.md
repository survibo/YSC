# 실험 결과

GPU 서버에서 돌린 결과 묶음. 압축을 풀면 `runs*/<run>/` 안에 `config.json`, `metrics.csv`, `summary.json`이 있다.

```bash
mkdir noise && tar xzf noise.tgz -C noise
python analyze.py "noise/runs_cell0.1/*"
```

- 일반화 경계: `boundary_map`, `wiring_boundary`, `wiring_commute`, `gpt_wiring`, `gpt_boundary`
- 칸당 관찰 수, coverage: `coverage_control`, `coverage_low`, `obs_prediction`, `answer_count_prediction`
- 필요한 예제 수: `law_*`, `examples_*`, `seq_length`, `weight_decay`, `gpt_examples`, `wd_range`
- 합성 규칙: `composition_*`
- 오답 암기: `noise*`
- 체크포인트 내부 분석: `checkpoint_*`
- 고전 grokking 과제, 무작위 연산표: `classic_and_random_table`
