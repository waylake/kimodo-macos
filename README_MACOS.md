# Kimodo on macOS (Apple Silicon)

This fork makes [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) — text-to-motion generation for humanoid characters —
build and run on Apple Silicon Macs, with no NVIDIA GPU and no Linux box.

Upstream targets Linux + CUDA. On a Mac the package does not even build: the bundled C++ motion-correction extension
hard-codes x86 SIMD (`-msse4.1 -mavx`, SSE/AVX intrinsics) and has two Clang-only compile errors. This fork fixes the
build, adds Apple GPU (MPS) selection, and adds an optional 8-bit text encoder so the whole pipeline fits comfortably in
unified memory.

Tested on: MacBook Pro M1 Max (64 GB), macOS 27, Python 3.12, PyTorch 2.14.1 (MPS), model `Kimodo-SOMA-RP-v1.2`.
Measured: a 4-second clip with BVH export in **27.5 s**, generator peak memory **2.6 GB** (text encoder running as a separate 8-bit service).

## What changed (vs. upstream `main`)

| File | Change |
|---|---|
| `MotionCorrection/CMakeLists.txt` | Apply `-msse4.1 -mavx` only on x86; plain flags on `arm64`/`aarch64` |
| `MotionCorrection/src/cpp/Math/SIMD.h` | On ARM, include [`sse2neon.h`](https://github.com/DLTcollab/sse2neon) (MIT) instead of `<immintrin.h>` — translates SSE intrinsics to NEON |
| `MotionCorrection/src/cpp/Math/sse2neon.h` | Vendored sse2neon header (MIT, unmodified) |
| `MotionCorrection/src/cpp/Math/Vector.inl` | `_mm_permutevar_ps` is AVX-only (not covered by sse2neon): scalar 4-lane permute on ARM, original intrinsic on x86 |
| `MotionCorrection/src/cpp/Compiler.h` | `FORCE_INLINE` was defined for MSVC and GCC but not Clang → define it for Clang too |
| `MotionCorrection/src/cpp/Math/Scalar.h` | Add missing `<stdlib.h>` (`labs`/`llabs`) — worked on Linux only via transitive includes |
| `kimodo/scripts/generate.py` | Device selection: `cuda` → **`mps`** → `cpu`; BVH export runs on CPU when the model is on MPS (the SOMA skeleton holds float64 buffers) |
| `kimodo/model/kimodo_model.py` | On MPS, cast float64 buffers (diffusion schedules) to float32 before `.to(device)` — MPS has no float64 |
| `kimodo/model/llm2vec/llm2vec_wrapper.py` | Optional `KIMODO_TE_8BIT=1`: weight-only int8 quantization of the LLM2Vec text encoder via `optimum-quanto` (~16 GB → ~8.5 GB) |

x86/CUDA behavior is unchanged: every change is either ARM-guarded or opt-in.

## Install

```bash
git clone https://github.com/waylake/kimodo-macos.git && cd kimodo-macos
uv venv --python 3.12 .venv && source .venv/bin/activate   # use 3.11+; see "Known issues"
uv pip install torch ".[all]" optimum-quanto
python -c "import kimodo, torch; print(torch.backends.mps.is_available())"   # True
```

(`pip` works too: `python -m venv .venv && pip install torch ".[all]" optimum-quanto`.)

## Run

Kimodo encodes the prompt with an LLM2Vec encoder built on **Meta Llama 3 8B Instruct**. Start the encoder service once,
then generate as many clips as you like:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
GRADIO_ALLOWED_PATHS=/tmp/text_encoder,/private/tmp/text_encoder \
  KIMODO_TE_8BIT=1 TEXT_ENCODER_DEVICE=mps kimodo_textencoder &     # serves on 127.0.0.1:9550

kimodo_gen "A person standing completely still slowly turns only the head toward the camera, then freezes." \
  --model Kimodo-SOMA-RP-v1.2 --duration 4 --num_samples 3 --bvh --output out/head_turn
```

`--bvh` writes BVH files you can retarget onto any humanoid rig (Blender, three.js `SkeletonUtils.retargetClip`, Unity,
Unreal, Godot).

### Text encoder weights

- **Default (unchanged from upstream):** the encoder pulls `meta-llama/Meta-Llama-3-8B-Instruct`, which is gated —
  request access on Hugging Face and `hf auth login`.
- **Local weights:** set `TEXT_ENCODERS_DIR=/path/to/encoders` and place the two LLM2Vec adapters there as
  `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` and `...-mntp-supervised`. Their `adapter_config.json` /
  `config.json` name the base model; point that to any Llama 3 8B Instruct copy you are licensed to use.

Either way, the Llama 3 weights are governed by the
[Meta Llama 3 Community License](https://llama.meta.com/llama3/license/); Kimodo model weights by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Generated motion clips contain neither model.

## Known issues

- **Python 3.10:** the SciPy wheels that support 3.10 (≤ 1.15) fail to load on recent macOS
  (`section '__DATA/__thread_bss' has a zero-fill section type`). Use Python 3.11+ (SciPy ≥ 1.16).
- Some ops fall back to CPU on MPS; keep `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- On macOS the encoder service writes to `/tmp/text_encoder`, which Gradio rejects (macOS temp is under `/var/folders`, and `/tmp` resolves to `/private/tmp`). Without `GRADIO_ALLOWED_PATHS` the client silently falls back to loading the full encoder in-process (~16 GB, every run).
- 8-bit encoding slightly shifts the text embeddings the diffusion model was trained on. Prefer full bf16 when memory
  allows (64 GB Macs can run it unquantized: omit `KIMODO_TE_8BIT`).

## License

Kimodo source: Apache-2.0 (see `LICENSE`, `ATTRIBUTIONS.MD`). This fork's changes are released under the same license.
`sse2neon.h`: MIT. Not affiliated with or endorsed by NVIDIA.

---

### 한국어 요약

엔비디아 Kimodo(문장 → 캐릭터 동작 생성)를 애플 칩 맥북에서 빌드·실행되게 고친 포크입니다. 원본은 리눅스+엔비디아 전용이라
맥에서는 C++ 부품 빌드부터 실패합니다(인텔 전용 명령, Clang 오류). 이 포크는 빌드를 고치고, 애플 칩 그래픽(MPS)을 쓰게 하고,
글 해석기(라마3 기반)를 8비트로 줄여 메모리를 절반으로 쓰는 선택 옵션을 추가했습니다. 인텔·엔비디아 환경의 동작은 그대로입니다.
