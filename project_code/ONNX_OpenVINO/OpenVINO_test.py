import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from openvino import Core
import os

# ============================================================
# 1. 설정
# ============================================================
ONNX_PATH = r"F:\\model\\model.onnx"
IR_PATH   = r"F:\\model\\model.xml"
SEQ_LEN = 128
MODEL_NAME = "distilbert-base-multilingual-cased" #distilbert-base-uncased"
N_ITERS = 50

# ============================================================
# 2. Tokenizer 준비 + int64 변환
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text = "I feel extremely tired and discouraged these days."

inputs = tokenizer(
    text,
    max_length=SEQ_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="np"
)

# ONNX Runtime은 int64를 요구함
input_ids = inputs["input_ids"].astype(np.int64)
attention_mask = inputs["attention_mask"].astype(np.int64)


# ============================================================
# 3. ONNX Runtime: CPUExecutionProvider
# ============================================================
session_cpu = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

def bench_ort_cpu():
    start = time.perf_counter()
    for _ in range(N_ITERS):
        session_cpu.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
    return (time.perf_counter() - start) / N_ITERS


# ============================================================
# 4. ONNX Runtime: OpenVINOExecutionProvider (설치 여부 체크!!)
# ============================================================
available_eps = ort.get_available_providers()
print("사용 가능 EP:", available_eps)

if "OpenVINOExecutionProvider" in available_eps:
    session_ov_ep = ort.InferenceSession(
        ONNX_PATH,
        providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    )

    def bench_ort_ov_ep():
        start = time.perf_counter()
        for _ in range(N_ITERS):
            session_ov_ep.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
        return (time.perf_counter() - start) / N_ITERS
else:
    bench_ort_ov_ep = None
    print("OpenVINOExecutionProvider 사용 불가")


# ============================================================
# 5. OpenVINO IR 변환
# ============================================================
print("ONNX → OpenVINO IR 변환 중...")

ovc_cmd = f'ovc "{ONNX_PATH}" --output_model "{IR_PATH}"'
print("실행:", ovc_cmd)
os.system(ovc_cmd)

if not os.path.exists(IR_PATH):
    raise FileNotFoundError(f"IR 파일 생성 실패: {IR_PATH}")


# ============================================================
# 6. OpenVINO Runtime (IR)
# ============================================================
core = Core()
compiled_ov = core.compile_model(IR_PATH, "CPU")
output_layer = compiled_ov.outputs[0]

def bench_ov_runtime():
    start = time.perf_counter()
    for _ in range(N_ITERS):
        compiled_ov({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })[output_layer]
    return (time.perf_counter() - start) / N_ITERS


# ============================================================
# 7. 속도 측정
# ============================================================
print("속도 측정 중...")

t_cpu = bench_ort_cpu()
print("CPU OK")

t_ep = bench_ort_ov_ep() if bench_ort_ov_ep else None
if t_ep is not None:
    print("OpenVINO EP OK")

t_ov = bench_ov_runtime()
print("OpenVINO Runtime OK")

# ============================================================
# 8. 출력
# ============================================================
print("\n====== 성능 비교 결과 (평균 per inference) ======")
print(f"1) ORT + CPU EP             : {t_cpu*1000:.3f} ms")

if t_ep is not None:
    print(f"2) ORT + OpenVINO EP        : {t_ep*1000:.3f} ms")
else:
    print("2) ORT + OpenVINO EP        : 사용 불가")

print(f"3) OpenVINO Runtime (IR)    : {t_ov*1000:.3f} ms")
print("====================================================")
