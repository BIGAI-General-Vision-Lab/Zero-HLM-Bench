"""Validate policy modality contracts before simulator control begins."""

from collections.abc import Mapping
from numbers import Integral


def modality_signature(configs):
    if not isinstance(configs, Mapping):
        raise ValueError("Policy modality configuration must be a mapping")
    result = {}
    for name, config in configs.items():
        if isinstance(config, Mapping):
            fields = config.get("as_json", config)
            if not isinstance(fields, Mapping):
                raise ValueError(f"Invalid serialized modality: {name}")
            keys = fields.get("modality_keys")
            indices = fields.get("delta_indices")
        else:
            keys = getattr(config, "modality_keys", None)
            indices = getattr(config, "delta_indices", None)
        if not isinstance(keys, (list, tuple)) or not keys or not all(isinstance(key, str) for key in keys):
            raise ValueError(f"Invalid modality_keys for {name}")
        if not isinstance(indices, (list, tuple)) or not indices or not all(
            isinstance(index, Integral) and not isinstance(index, bool) for index in indices
        ):
            raise ValueError(f"Invalid delta_indices for {name}")
        result[name] = {"modality_keys": list(keys), "delta_indices": list(indices)}
    return result


def validate_policy_modalities(expected, received):
    expected_signature = modality_signature(expected)
    received_signature = modality_signature(received)
    if expected_signature == received_signature:
        return
    differences = []
    for name in sorted(set(expected_signature) | set(received_signature)):
        if expected_signature.get(name) != received_signature.get(name):
            differences.append(
                f"{name}: expected {expected_signature.get(name)!r}, received {received_signature.get(name)!r}"
            )
    raise ValueError("Policy server modality contract differs from the task profile: " + "; ".join(differences))
