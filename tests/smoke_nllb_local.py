import argparse
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="NLLB local smoke test")
    parser.add_argument("--text", default="Hello, this is a smoke test for NLLB local translation.")
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="es")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pyvt_root = root / "pyvideotrans"
    if str(pyvt_root) not in sys.path:
        sys.path.insert(0, str(pyvt_root))

    print("=== NLLB Local Smoke Test ===")
    print(f"project_root={root}")
    print(f"pyvideotrans_root={pyvt_root}")

    try:
        import torch

        print(f"torch_version={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        expected_dtype = "torch.float16" if torch.cuda.is_available() else "torch.float32"
        print(f"expected_nllb_dtype={expected_dtype}")
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            print(f"cuda_device_index={idx}")
            print(f"cuda_device_name={torch.cuda.get_device_name(idx)}")
            print(f"cuda_capability={torch.cuda.get_device_capability(idx)}")
    except Exception:
        print("torch_probe=FAIL")
        traceback.print_exc()
        return 2

    try:
        from videotrans.translator._nllb200 import NLLB200Trans

        t = NLLB200Trans(
            translate_type=2,
            text_list=[],
            source_code=args.source,
            target_code=args.target,
            target_language_name="Spanish" if args.target == "es" else args.target,
            is_test=True,
        )

        print("warmup_start=1")
        t._download()
        print("warmup_ok=1")
        print(f"nllb_device={getattr(t, 'device', None)}")

        dtype = "unknown"
        model = getattr(t, "model", None)
        if model is not None:
            try:
                dtype = str(next(model.parameters()).dtype)
            except Exception:
                pass
        print(f"nllb_dtype={dtype}")

        translated = t._item_task(args.text)
        print(f"translate_input={args.text}")
        print(f"translate_output={translated}")

        t._unload()
        print("warmup_unload_ok=1")
        print("smoke_result=PASS")
        return 0
    except Exception as exc:
        print("smoke_result=FAIL")
        print(f"error_type={type(exc).__name__}")
        print(f"error_msg={exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
