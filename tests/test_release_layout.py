"""Release entrypoints and diagnostic defaults follow the published task profiles."""

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime/scripts/imitation_learning/robomimic'


def test_profiles_reference_descriptive_task_entrypoints():
    for number in (1, 2, 3):
        config = json.loads((ROOT / f'configs/task{number}.json').read_text())
        path = ROOT / 'runtime' / config['evaluator']
        assert path.name == f'evaluate_task{number}.py'
        assert path.is_file()
        assert config['args']['--diagnostic_video_interval'] == 10
        assert not any('DIAGNOSTIC_VIDEO_INTERVAL' in key for key in config['env'])


def test_diagnostic_defaults_cover_the_released_manifest():
    paths = list(RUNTIME.glob('task1*contract.py'))
    assert len(paths) == 1
    spec = importlib.util.spec_from_file_location('release_layout_contract', paths[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        config = json.loads((ROOT / 'configs/task1.json').read_text())
        assert module.TASK1_SCENARIO_COUNT == config['manifest']['rows']
        assert module.diagnostic_video_indices() == tuple(range(9, config['manifest']['rows'], 10))
    finally:
        sys.modules.pop(spec.name)
