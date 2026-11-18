# /metric — 통합 평가 파이프라인

RRG(Radiology Report Generation) baseline 하나를 평가할 때, 이 폴더 하나로 6개 metric
(RadGraph-F1, CheXbert-F1, RadCliQ-v1, RaTEScore, GREEN, RadFact-F1)을 **한 줄 명령어**로
돌릴 수 있습니다. 새 baseline을 추가할 때는 아래 input contract만 맞추면 되고, 이 폴더의
코드는 고칠 필요가 없습니다.

**CRIMSON은 7번째 metric이지만 아직 `run_eval.sh`에 안 묶여있어서 `metric/crimson/`에서
따로 돌려야 합니다** — 아래 "CRIMSON만 따로 알아둘 것" 참고.

## 0. 최초 1회 환경 설정

세 개의 conda env가 필요합니다 (`run_eval.sh`가 자동으로 올바른 env를 골라 실행하므로,
사용자는 `radeval`만 activate하고 있으면 됩니다):

```bash
# 1) 메인 env: radgraph/f1chexbert/radcliq/green 전부 + run_eval.sh 실행
conda create -n radeval python=3.11 -y
conda activate radeval
pip install -e /home/seona/prj-brl/metric/RadEval

# 2) RaTEScore 전용 env (radeval와 numpy 버전이 충돌해서 분리 — 아래 "왜 env가 3개" 참고)
conda create -n radeval-rate python=3.11 -y
conda activate radeval-rate
pip install -e /home/seona/prj-brl/metric/RadEval
pip install "numpy<2.0"

# 3) RadFact-F1 전용 env (vLLM이 자기 torch/CUDA 빌드를 요구해서 분리)
conda create -n vllm-decompose python=3.11 -y
conda activate vllm-decompose
pip install vllm
```

## Input contract

평가하려는 baseline의 `<run_dir>/predictions.json`이 아래 형태의 JSON list여야 합니다:

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

- `base_model/MedGemma/generate_reports.py`는 이 형식으로 **직접** 생성합니다.
- 다른 코드베이스(R2GenGPT 등)는 자기 고유 출력 포맷을 쓰므로, `convert_results.py` 같은
  작은 변환 스크립트로 이 포맷으로 바꿔줍니다 (`baseline_rrg/R2GenGPT/convert_results.py`
  참고 — 새 baseline 추가 시 이 패턴을 그대로 따라 작성하면 됩니다).

## 사용법

```bash
conda activate radeval
cd /home/seona/prj-brl/metric

# 전체 6개 metric (기본값)
./run_eval.sh <run_dir> --dataset mimic-cxr --split test

# metric 선택
./run_eval.sh <run_dir> --dataset mimic-cxr --split test --metrics radgraph,f1chexbert,radcliq

# GPU 지정 (radeval 쪽 metric들에 적용; radfact는 --radfact_tp로 별도 GPU 개수 지정)
./run_eval.sh <run_dir> --dataset mimic-cxr --split test --gpus 0,2,3
./run_eval.sh <run_dir> --dataset mimic-cxr --split test --radfact_tp 4
```

`radeval` 하나만 activate한 채로 실행하면 됩니다 — `ratescore`는 내부에서
`conda run -n radeval-rate`로, RadFact-F1은 `conda run -n vllm-decompose`로 자동
전환되어 실행됩니다 (env를 직접 바꿀 필요 없음).

결과는 `<run_dir>/scores.json`에 전부 모입니다 (여러 번 다른 `--metrics`로 돌려도 기존
점수를 덮어쓰지 않고 병합됩니다). per-sample 점수는 `<run_dir>/predictions.json`의 각
레코드에 `radeval_scores` 필드로 합쳐집니다.

여러 baseline을 돌린 뒤 한 표로 비교하려면 `python3 collect_results.py`
(→ `/home/seona/prj-brl/results/RRG_RESULTS.md` 생성/갱신, `ROW_TO_RUN_DIR`에 run_dir
한 줄만 등록하면 됨).

## RadFact-F1만 따로 알아둘 것

공식 `radeval` 패키지의 RadFact는 `"radfact_ct"` 하나뿐이고 CT 모달리티 + OpenAI API
전용이라, CXR(흉부 X-ray)용은 `radfact/`에 로컬 LLM(Llama-3-70B-Instruct) 기반으로
자체 구현했습니다. 세 단계로 이루어집니다:

1. **GT findings decompose** — dataset+split 하나당 1번만 하면 되는 공용 작업(모든
   baseline이 재사용). **`run_eval.sh`가 자동으로 하지 않습니다** — 캐시
   (`radfact/gt_cache/<dataset>_<split>_gt_radfact_findings.json`)가 없으면 아래
   명령어를 안내하고 멈춥니다:
   ```bash
   conda activate vllm-decompose
   cd /home/seona/prj-brl/metric/radfact
   CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/decompose_radfact_vllm.py \
       --annotation_path /mnt/.cache/huggingface/datasets/mimic-cxr/annotation.json \
       --split test --output_file gt_cache/mimic-cxr_test_gt_radfact_findings.json \
       --tensor_parallel_size 4
   ```
   mimic-cxr의 train/val/test는 이미 캐시가 만들어져 있습니다 (`radfact/gt_cache/`).
2. **candidate decompose + NLI 비교** — `run_eval.sh`가 자동으로 수행 (vLLM 버전 스크립트,
   `conda run -n vllm-decompose`로 전환). 캐시가 이미 있으면(`<run_dir>/radfact_candidate_findings.json`,
   `<run_dir>/radfact_scores.json`) 재실행하지 않고 건너뜁니다 — 다시 하고 싶으면 그 파일을
   지우면 됩니다.
3. **집계** — `radfact/fold_radfact_scores.py`가 원본 RadFact 논문의 집계 공식(mean(P),
   mean(R)을 먼저 구하고 그 조화평균을 F1으로 — per-sample F1의 평균이 아님)으로
   `scores.json`에 `radfact_precision/recall/f1`을 씁니다. candidate/GT 양쪽의 decompose
   상태(`status`)가 둘 다 `"ok"`인 샘플만 집계에 포함됩니다.

LLM이 JSON을 가끔 깨뜨려서(`status: parse_error`) 일부 report는 decompose가 실패할 수
있습니다 (처음엔 이 때문에 ~15% 가량 집계에서 제외됐었음). 복구하려면:
```bash
bash /home/seona/prj-brl/metric/radfact/run_retry_all_and_refold.sh
```
(실패한 것만 `temperature=0.3`로 재시도 — greedy로는 같은 실수가 반복돼서 거의 안 고쳐짐.
몇 번 더 돌려도 되지만 100% 복구를 목표로 할 필요는 없음 — 줄어드는 폭이 작아지면 멈추고
남은 건 집계에서 제외하면 됨.)

## CRIMSON만 따로 알아둘 것

CRIMSON(Baharoon et al. 2026, arXiv:2603.06183)도 `radeval` 패키지에 없는 metric이라
`crimson/`에 자체 구현했습니다. 기본 모델은 `rajpurkarlab/medgemma-4b-it-crimson`
(HF, ~8GB bf16), patient_context(나이/indication)는 안 넘깁니다 — MIMIC-CXR/IU X-Ray
둘 다 그 정보가 없어서.

폴더에 스크립트가 2개 있는데 **`crimson_vllm.py`만 쓰세요**:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-decompose   # CRIMSON 전용 env가 아니라 radfact와 같은 env로 돌림
cd /home/seona/prj-brl/metric/crimson

CUDA_VISIBLE_DEVICES=<비어있는 GPU 1개> python3 crimson_vllm.py <run_dir> [run_dir2 ...]
```

- **`crimson.py`(순차 HF `pipeline()` 버전)는 쓰지 마세요** — CRIMSON 라이브러리의
  단일 샘플용 `evaluate()`는 LLM이 JSON을 깨뜨리면 예외를 안 잡고 그대로 죽는데,
  run_dir 하나(수천 개)가 통째로 끝나야만 저장하는 구조라 중간에 죽으면 그 run_dir에서
  처리한 거 전부 날아갑니다 (실제로 겪은 사고). `crimson_vllm.py`는 같은 라이브러리의
  배치용 `evaluate_batch()`를 쓰는데, 이쪽은 샘플 하나 파싱 실패해도 그 샘플만 건너뛰고
  계속 진행합니다.
- `crimson_vllm.py`는 500개마다 `<run_dir>/evidence/crimson_partial.json`에 체크포인트
  — 중단돼도 재실행하면 거기서부터 이어서 함.
- `scores.json`에 이미 `crimson` 키가 있는 run_dir은 자동으로 skip (여러 run_dir을 한
  명령에 같이 넣어도 안전 — 끝난 건 건너뛰고 안 끝난 것만 처리).
- 4B라 GPU 1개로 충분합니다 (vLLM이 `CUDA_VISIBLE_DEVICES`로 보이는 GPU에만 올라감 —
  radfact처럼 `--tensor_parallel_size`로 여러 GPU에 쪼갤 필요 없음).
- 일부 샘플은 파싱 실패할 수 있는데 `scores.json`의 `crimson` 리스트에 `NaN`으로
  기록됩니다 — `collect_results.py`의 `mean_of()`가 NaN을 제외하고 평균 냅니다.

## 왜 env가 3개인가 (radeval / radeval-rate / vllm-decompose)

- **radeval vs radeval-rate**: GREEN이 쓰는 `datasets`/`pyarrow`는 numpy≥2.0을 요구하는데,
  RaTEScore가 쓰는 `thinc`(spacy 백엔드)의 컴파일된 C 확장은 numpy<2.0용으로 빌드돼 있어서
  같은 env에 둘 다 넣을 수 없음 — ratescore만 numpy<2.0인 별도 env(`radeval-rate`)로 분리.
- **vllm-decompose**: vLLM이 자기 전용 torch/CUDA 빌드를 pin하기 때문에 radeval와 공존 불가 —
  RadFact-F1(Llama-3-70B-Instruct 추론)만 이 env로 분리.

`run_eval.sh`는 이 셋을 다 알고 있어서, 사용자는 `radeval`만 activate하면 됩니다.

## 폴더 구조

```
metric/
  README.md                  # 이 파일
  run_eval.sh                 # 통합 entrypoint
  collect_results.py           # 여러 baseline의 scores.json을 모아 results/RRG_RESULTS.md 생성
  RadEval/                    # 공식 radeval 패키지 소스 (pip install -e .로 설치된 원본, 직접 수정 X)
  radeval_metrics/             # metric 하나당 파일 하나, 독립적으로 실행 가능
    common.py                  # 공유 I/O (predictions.json 읽기/쓰기, scores.json 병합)
    report_cleaning.py         # GT의 NLM 비식별화 placeholder("XXXX") 제거
    radgraph.py / f1chexbert.py / radcliq.py / ratescore.py / green.py
  radfact/                    # RadFact-F1 (CXR) — radeval 패키지엔 CT용(radfact_ct)만 있어서 자체 구현
    decomposition/            # 핵심 라이브러리 (prompt 생성, 파싱, NLI 로직)
    RadEval/                  # 원본 microsoft/RadFact clone (프롬프트 에셋만 읽어다 씀, 코드는 import 안 함)
    scripts/
      decompose_radfact_vllm.py             # GT report → finding 분해 (vLLM, --retry_failed 지원)
      decompose_radfact_predictions_vllm.py # 생성 report → finding 분해 (vLLM, --retry_failed 지원)
      run_eval_radfact_vllm.py              # 둘을 NLI로 비교 → precision/recall/F1 (vLLM)
      run_decompose_radfact.py / run_decompose_radfact_predictions.py / run_eval_radfact.py
                                             # 순차(non-vLLM) 버전 -- 느림(수십 시간), 지금은 안 씀
    fold_radfact_scores.py    # radfact_scores.json → scores.json (원본 집계 공식 + 오염 샘플 제외)
    run_retry_all_and_refold.sh  # train/test GT + candidate의 parse_error 전부 재시도 후 재집계
    gt_cache/                  # GT decompose 결과 캐시 (dataset+split당 1번만, baseline 간 재사용)
  crimson/                    # CRIMSON (radeval 패키지에 없어서 자체 구현)
    crimson_vllm.py             # 이것만 쓸 것 (vLLM, 체크포인트/skip 지원)
    crimson.py                  # 쓰지 말 것 — 파싱 실패 시 run_dir 전체가 죽음 (위 섹션 참고)
```

## Metric별 모델 다운로드 현황

| Metric | 구현체 | 필요 모델 |
|---|---|---|
| RadGraph-F1 | `radeval` 패키지 | radgraph-xl |
| CheXbert-F1 | `radeval` 패키지 | chexbert.pth |
| RadCliQ | `radeval` 패키지 | (위 2개 + distilroberta-base) |
| RaTEScore | `radeval` 패키지 | `Angelakeke/RaTE-NER-Deberta` + `FremyCompany/BioLORD-2023-C` |
| GREEN | `radeval` 패키지 | `StanfordAIMI/GREEN-radllama2-7b` |
| RadFact-F1 | 자체 구현 (`radfact/`) | `meta-llama/Meta-Llama-3-70B-Instruct` (132GB bf16) — RadFact 논문이 실제 entailment에 쓴 모델과 동일 |
| CRIMSON | 자체 구현 (`crimson/`) | `rajpurkarlab/medgemma-4b-it-crimson` (~8GB bf16) |

전부 `/mnt/.cache/huggingface/models/`에 캐시되어 있습니다 (팀 공유 캐시) — CRIMSON
모델만 예외로 기본 HF 캐시(`~/.cache/huggingface/hub/`)에 있습니다.

## 데이터셋 위치

MIMIC-CXR/IU X-Ray 실제 데이터(이미지+annotation.json)는
`/mnt/.cache/huggingface/datasets/{mimic-cxr,iu_xray}/`에 있습니다 (팀 공유 캐시 —
`/home/seona/prj-brl/dataset/`에는 다운로드 스크립트/로그만 남아있고 실제 데이터는 없음).
