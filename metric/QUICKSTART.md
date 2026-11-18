# Quickstart — RRG 베이스라인 평가 (처음부터 끝까지)

baseline을 평가하고 싶을 때, 이 파일에 적힌 순서만 따라오면 됩니다.
"왜 이렇게 돼있는지"는 안 다루고, "뭘 받고 뭘 실행하면 끝나는지"만 적어뒀습니다 —
배경 설명이 필요하면 `README.md`

이 서버(공유 캐시 `/mnt/.cache/huggingface/`)에서 작업한다는 전제입니다 — 모델/데이터셋은
이미 전부 다운로드되어 있어서 **새로 받을 파일은 없습니다.**

## 0. 필요한 것 (한 번만 확인)

| 종류 | 위치 | 새로 받아야 하나? |
|---|---|---|
| 평가 코드 | `/home/seona/prj-brl/metric/` (이 폴더) | 아니오, 이미 있음 |
| 모델들 (RadGraph-xl, CheXbert, RaTEScore, GREEN, Llama-3-70B, medgemma-4b-it-crimson) | `/mnt/.cache/huggingface/models/` | 아니오, 공유 캐시에 이미 있음 |
| MIMIC-CXR / IU X-Ray 데이터셋 | `/mnt/.cache/huggingface/datasets/{mimic-cxr,iu_xray}/` | 아니오, 공유 캐시에 이미 있음 |
| conda env 3개 (`radeval`, `radeval-rate`, `vllm-decompose`) | — | **예, 아래 1단계에서 설치** (사람마다 conda env는 개인 소유라 공유 안 됨) |


## 1. conda env 설치 (최초 1회만)

```bash
git clone https://github.com/jbdel/RadEval.git

conda create -n radeval python=3.11 -y
conda activate radeval
pip install -e ./RadEval

conda create -n radeval-rate python=3.11 -y
conda activate radeval-rate
pip install -e ./RadEval
pip install "numpy<2.0"

conda create -n vllm-decompose python=3.11 -y
conda activate vllm-decompose
pip install vllm
```

(`radeval-rate`는 RaTEScore 전용, `vllm-decompose`는 RadFact-F1 + CRIMSON 전용 — 둘 다
`radeval` 하나만 activate해서 쓰면 내부에서 자동으로 전환되니, 직접 activate할 일은
거의 없습니다. CRIMSON은 `vllm-decompose` env를 그대로 재사용하고 따로 설치하는 게
없습니다.)

## 2. baseline의 `predictions.json` 만들기

평가하려는 run의 결과를 아래 형태의 JSON list 하나로 만들어서 `<run_dir>/predictions.json`에
저장하기 (`<run_dir>`은 아무 경로나 상관없음, 예: `baseline_rrg/MyModel/results/mymodel_mimic_test/`):

```json
[
  {
    "id": "02aa804e-...",
    "study_id": "02aa804e-...",
    "gt_report": "The heart size and pulmonary vascularity ...",
    "generated_report": "The lungs are clear bilaterally ..."
  },
  ...
]
```

(다른 포맷으로 이미 결과를 뽑아놨다면, 이 포맷으로 바꿔주는 작은 변환 스크립트 하나만 있으면 됨)

## 3. 평가 실행

```bash
conda activate radeval
cd ./metric

# RadGraph-F1, CheXbert-F1, RadCliQ-v1, RaTEScore, GREEN, RadFact-F1 (6개 전부)
./run_eval.sh <run_dir> --dataset mimic-cxr --split test
```

CRIMSON(7번째 metric)은 아직 `run_eval.sh`에 안 묶여서 따로 돌립니다:

```bash
conda activate vllm-decompose
cd ./metric/crimson

# 비어있는 GPU 1개만 쓰면 됨 (4B 모델이라 1개로 충분)
CUDA_VISIBLE_DEVICES=<비어있는 GPU 번호> python3 crimson_vllm.py <run_dir>
```

끝나면 둘 다 같은 파일에 모임: **`<run_dir>/scores.json`** (여러 번 나눠 돌려도
기존 점수를 덮어쓰지 않고 병합됨).

## 4. 결과표에 반영

```bash
cd ./metric
python3 collect_results.py
```

→ (선택) `./results/RRG_RESULTS.md`가 갱신됩니다. baseline이 표에
안 보이면, `collect_results.py` 안의 해당 `*_ROW_TO_RUN_DIR` 딕셔너리에 `"행 이름": Path(...)`
한 줄만 추가하면 됨

## 막히면

- GPU 메모리/프로세스 상태: `nvidia-smi`
- 각 단계가 정확히 뭘 하는지, env가 왜 3개인지, RadFact/CRIMSON 내부 동작, 파싱 실패 재시도
  방법 등은 → `README.md`
