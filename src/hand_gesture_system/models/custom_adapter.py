from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.models.prediction_utils import best_prediction_from_output
from hand_gesture_system.models.sequence_base import SequenceGestureModelAdapter
from hand_gesture_system.types import GesturePrediction

# NOTE: This module is the framework layer for plugging user AI models into the
# pipeline without changing core inference code. The user supplies an entrypoint:
#   - module.path:Symbol
#   - /path/to/file.py:Symbol
# The symbol may be:
#   - a class (instantiated once),
#   - a factory function (called once),
#   - a callable object.
#
# Static adapters look for:
#   predict(feature_vector)
#
# Sequence adapters look for:
#   predict_sequence(sequence_features)
#
# Both also accept plain callables returning model output.
# Returned output can be GesturePrediction objects, dict[label->score], or raw
# logits/probabilities/class-id that get normalized via prediction_utils.


def load_custom_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom model config file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            "Custom model config must be a JSON object (key/value map)."
        )
    return data


def _load_module_from_file(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Custom model file not found: {path}")

    module_name = f"_hand_gesture_custom_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec from file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_symbol(entrypoint: str) -> Any:
    if ":" not in entrypoint:
        raise ValueError(
            "Custom entrypoint must be in format 'module.path:Symbol' "
            "or '/path/to/file.py:Symbol'"
        )

    module_ref, symbol_name = entrypoint.split(":", 1)
    module_ref = module_ref.strip()
    symbol_name = symbol_name.strip()
    if not module_ref or not symbol_name:
        raise ValueError(
            "Custom entrypoint must include both module/file and symbol name."
        )

    is_file_ref = (
        module_ref.endswith(".py")
        or "/" in module_ref
        or "\\" in module_ref
        or module_ref.startswith(".")
    )

    if is_file_ref:
        module = _load_module_from_file(Path(module_ref).expanduser())
    else:
        module = importlib.import_module(module_ref)

    if not hasattr(module, symbol_name):
        raise AttributeError(f"Symbol '{symbol_name}' not found in '{module_ref}'")
    return getattr(module, symbol_name)


def _call_with_supported_kwargs(factory: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(factory)
    parameters = signature.parameters
    has_var_kwargs = any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in parameters.values()
    )

    if has_var_kwargs:
        return factory(**kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in parameters}
    return factory(**filtered)


def _instantiate_custom_model(
    entrypoint: str,
    *,
    model_path: str | None,
    labels: list[str] | None,
    config: dict[str, Any] | None,
) -> Any:
    symbol = _load_symbol(entrypoint)

    constructor_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "labels": labels,
        "config": config or {},
    }

    if inspect.isclass(symbol):
        return _call_with_supported_kwargs(symbol, constructor_kwargs)

    if callable(symbol):
        created = _call_with_supported_kwargs(symbol, constructor_kwargs)
        return created

    return symbol


def _normalize_custom_output(
    output: Any,
    *,
    labels: list[str] | None,
    source: str,
) -> list[GesturePrediction]:
    if output is None:
        return []

    if isinstance(output, GesturePrediction):
        return [output]

    if isinstance(output, tuple):
        output = list(output)

    if isinstance(output, list):
        if not output:
            return []
        if all(isinstance(item, GesturePrediction) for item in output):
            return output

    if isinstance(output, dict):
        predictions: list[GesturePrediction] = []
        for label, score in sorted(
            output.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            predictions.append(
                GesturePrediction(
                    label=str(label),
                    score=float(score),
                    source=source,
                )
            )
        return predictions

    return best_prediction_from_output(output, labels=labels, source=source)


class CustomGestureAdapter(GestureModelAdapter):
    """Adapter for user-defined static models loaded from Python symbols."""

    def __init__(
        self,
        entrypoint: str,
        *,
        model_path: str | None = None,
        labels: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        model_object = _instantiate_custom_model(
            entrypoint,
            model_path=model_path,
            labels=labels,
            config=config,
        )
        self._labels = labels
        self._source = "custom"

        if hasattr(model_object, "predict") and callable(model_object.predict):
            self._predict_fn = model_object.predict
            return

        if callable(model_object):
            self._predict_fn = model_object
            return

        raise TypeError(
            "Custom static model must be callable or implement 'predict(feature_vector)'."
        )

    def predict(self, feature_vector: np.ndarray) -> list[GesturePrediction]:
        output = self._predict_fn(np.asarray(feature_vector, dtype=np.float32))
        return _normalize_custom_output(
            output,
            labels=self._labels,
            source=self._source,
        )


class CustomSequenceGestureAdapter(SequenceGestureModelAdapter):
    """Adapter for user-defined temporal models loaded from Python symbols."""

    def __init__(
        self,
        entrypoint: str,
        *,
        model_path: str | None = None,
        labels: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        model_object = _instantiate_custom_model(
            entrypoint,
            model_path=model_path,
            labels=labels,
            config=config,
        )
        self._labels = labels
        self._source = "custom_sequence"

        if hasattr(model_object, "predict_sequence") and callable(
            model_object.predict_sequence
        ):
            self._predict_fn = model_object.predict_sequence
            return

        if callable(model_object):
            self._predict_fn = model_object
            return

        raise TypeError(
            "Custom sequence model must be callable or implement "
            "'predict_sequence(sequence_features)'."
        )

    def predict_sequence(self, sequence_features: np.ndarray) -> list[GesturePrediction]:
        output = self._predict_fn(np.asarray(sequence_features, dtype=np.float32))
        return _normalize_custom_output(
            output,
            labels=self._labels,
            source=self._source,
        )
