from pathlib import Path

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.whisper_base_en.export import (
  ALL_COMPONENTS,
  export_model,
)

import qai_hub as hub
hub.get_devices()

cwd = Path(__file__).parent
result = export_model(
  device='Snapdragon X Elite CRD',
  components=ALL_COMPONENTS,
  skip_profiling=True,
  skip_inferencing=True,
  output_dir=str(cwd / 'build' / 'whisper_base_en'),
  target_runtime=TargetRuntime.ONNX,
)
print(result)
