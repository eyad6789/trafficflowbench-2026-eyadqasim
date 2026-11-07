from pathlib import Path
import os
RELEASE = Path(os.environ["TFB_DATA_ROOT"])
OUT = Path(os.environ.get("TFB_OUTPUT_ROOT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
THREADS = int(os.environ.get("TFB_THREADS", "4"))
