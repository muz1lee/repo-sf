"""Object proposal helpers for real-to-sim extraction."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import requests
import yaml
from PIL import Image


@dataclass(frozen=True)
class ObjectProposal:
    label: str
    object_id: str
    bbox_xyxy: tuple[int, int, int, int] | None = None
    point_xy: tuple[int, int] | None = None
    confidence: float = 1.0
    mass_kg: float = 0.25
    friction: float = 0.8
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.bbox_xyxy is not None:
            data["bbox_xyxy"] = list(self.bbox_xyxy)
        if self.point_xy is not None:
            data["point_xy"] = list(self.point_xy)
        return data


def load_object_proposals(path: str | Path, *, image_size: tuple[int, int]) -> list[ObjectProposal]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = data.get("objects", data if isinstance(data, list) else [])
    if not isinstance(items, list):
        raise ValueError("object proposal YAML must contain a list or an 'objects' list")
    return [normalize_object_proposal(item, image_size=image_size, source="yaml") for item in items]


class QwenProposalClient:
    """OpenAI-compatible Qwen-VL object proposal client.

    The deployment details vary across servers, so this client only assumes the
    common `/chat/completions` shape and a JSON response in the assistant text.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        _load_env_file(os.environ.get("REAL2SIM_QWEN_ENV_FILE") or os.environ.get("QWEN_ENV_FILE"))
        self._base_url = (
            base_url
            or os.environ.get("QWEN_PROXY_URL")
            or os.environ.get("QWEN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or ("https://dashscope.aliyuncs.com/compatible-mode/v1" if os.environ.get("DASHSCOPE_API_KEY") else "")
        ).rstrip("/")
        self._api_key = (
            api_key
            or os.environ.get("QWEN_PROXY_KEY")
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self._model = model or os.environ.get("QWEN_MODEL") or "qwen3-vl-235b-a22b-instruct"
        self._timeout_s = float(timeout_s)
        if not self._base_url:
            raise ValueError("QwenProposalClient requires QWEN_PROXY_URL/QWEN_BASE_URL or explicit base_url")
        if not self._api_key:
            raise ValueError("QwenProposalClient requires QWEN_PROXY_KEY/QWEN_API_KEY/DASHSCOPE_API_KEY/OPENAI_API_KEY or explicit api_key")

    def propose(self, image_path: str | Path, *, target_labels: Sequence[str] | None = None) -> list[ObjectProposal]:
        image = Image.open(image_path).convert("RGB")
        prompt = _proposal_prompt(target_labels)
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    ],
                }
            ],
            "temperature": 0,
        }
        if os.environ.get("QWEN_PROXY_URL") and self._base_url == os.environ.get("QWEN_PROXY_URL", "").rstrip("/"):
            payload["backend"] = os.environ.get("QWEN_PROXY_BACKEND", "dashscope")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        content = _message_text(response.json()["choices"][0]["message"]["content"])
        data = _parse_json_from_text(content)
        items = data.get("objects", data if isinstance(data, list) else [])
        if not isinstance(items, list):
            raise ValueError(f"Qwen proposal response must contain object list, got: {data}")
        proposals = [normalize_object_proposal(item, image_size=image.size, source="qwen") for item in items]
        return _dedupe_object_ids(proposals)


def normalize_object_proposal(data: dict[str, Any], *, image_size: tuple[int, int], source: str) -> ObjectProposal:
    if not isinstance(data, dict):
        raise ValueError(f"object proposal must be a mapping, got {type(data).__name__}")
    label = str(data.get("label") or data.get("name") or data.get("object") or "").strip()
    if not label:
        raise ValueError(f"object proposal missing label: {data}")
    object_id = str(data.get("object_id") or data.get("id") or _slugify(label)).strip()
    bbox = _read_bbox(data, image_size=image_size, source=source)
    point = _read_point(data, image_size=image_size, source=source)
    return ObjectProposal(
        label=label,
        object_id=_slugify(object_id),
        bbox_xyxy=bbox,
        point_xy=point,
        confidence=float(data.get("confidence", 1.0)),
        mass_kg=float(data.get("mass_kg", 0.25)),
        friction=float(data.get("friction", 0.8)),
        source=source,
    )


def _proposal_prompt(target_labels: Sequence[str] | None) -> str:
    target = ", ".join(target_labels or [])
    target_text = f"Focus on these target objects if present: {target}." if target else "Find movable tabletop objects."
    return (
        "Return only valid JSON with schema "
        '{"objects":[{"label":str,"bbox_2d":[x0,y0,x1,y1],"point_2d":[x,y],"confidence":float,'
        '"mass_kg":float,"friction":float}]}. '
        "Use pixel coordinates in the input image. Make each bbox tight around one physical object. "
        "Choose point_2d near the visible center of that object, not on neighboring objects. "
        f"{target_text}"
    )


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _parse_json_from_text(text: str) -> Any:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    else:
        starts = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
        if starts:
            start = min(starts)
            end = max(stripped.rfind("}"), stripped.rfind("]"))
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _read_bbox(data: dict[str, Any], *, image_size: tuple[int, int], source: str) -> tuple[int, int, int, int] | None:
    raw = data.get("bbox_xyxy") or data.get("box_xyxy") or data.get("bbox_2d") or data.get("bbox") or data.get("box")
    if raw is None:
        return None
    if len(raw) != 4:
        raise ValueError(f"bbox must have four numbers: {raw}")
    width, height = image_size
    x0, y0, x1, y1 = _scale_coordinates([float(v) for v in raw], image_size, source=source)
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width - 1, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox must have positive area after clipping: {raw}")
    return x0, y0, x1, y1


def _read_point(data: dict[str, Any], *, image_size: tuple[int, int], source: str) -> tuple[int, int] | None:
    raw = data.get("point_xy") or data.get("point_2d") or data.get("point")
    if raw is None and data.get("points"):
        raw = data["points"][0]
    if raw is None:
        return None
    if len(raw) != 2:
        raise ValueError(f"point must have two numbers: {raw}")
    width, height = image_size
    x, y = _scale_coordinates([float(v) for v in raw], image_size, source=source)
    x, y = int(round(x)), int(round(y))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _scale_coordinates(values: list[float], image_size: tuple[int, int], *, source: str) -> list[float]:
    max_value = max(abs(float(value)) for value in values)
    width, height = image_size
    if max_value <= 1.0:
        return [
            float(value) * (width if index % 2 == 0 else height)
            for index, value in enumerate(values)
        ]
    if source == "qwen" and max_value <= 1000.0:
        return [
            float(value) / 1000.0 * (width if index % 2 == 0 else height)
            for index, value in enumerate(values)
        ]
    if max_value > max(width, height) and max_value <= 1000.0:
        return [
            float(value) / 1000.0 * (width if index % 2 == 0 else height)
            for index, value in enumerate(values)
        ]
    return [float(value) for value in values]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return str(content)


def _load_env_file(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value).expanduser()
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _dedupe_object_ids(proposals: list[ObjectProposal]) -> list[ObjectProposal]:
    seen: dict[str, int] = {}
    result: list[ObjectProposal] = []
    for proposal in proposals:
        count = seen.get(proposal.object_id, 0)
        seen[proposal.object_id] = count + 1
        if count == 0:
            result.append(proposal)
            continue
        result.append(
            ObjectProposal(
                label=proposal.label,
                object_id=f"{proposal.object_id}_{count + 1}",
                bbox_xyxy=proposal.bbox_xyxy,
                point_xy=proposal.point_xy,
                confidence=proposal.confidence,
                mass_kg=proposal.mass_kg,
                friction=proposal.friction,
                source=proposal.source,
            )
        )
    return result


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "object"
