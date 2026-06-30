import base64
import io
import json

from PIL import Image

from real2sim_scene_foundry.proposals import ObjectProposal, QwenProposalClient, load_object_proposals


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_load_object_proposals_normalizes_yaml_bbox_and_point(tmp_path):
    path = tmp_path / "objects.yaml"
    path.write_text(
        """
objects:
  - label: red cup
    object_id: cup_1
    bbox_xyxy: [10, 20, 40, 70]
    point_xy: [25, 45]
    mass_kg: 0.12
    friction: 0.7
""",
        encoding="utf-8",
    )

    proposals = load_object_proposals(path, image_size=(100, 80))

    assert proposals == [
        ObjectProposal(
            label="red cup",
            object_id="cup_1",
            bbox_xyxy=(10, 20, 40, 70),
            point_xy=(25, 45),
            confidence=1.0,
            mass_kg=0.12,
            friction=0.7,
            source="yaml",
        )
    ]


def test_qwen_proposal_client_posts_image_and_parses_json(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), color=(1, 2, 3)).save(image)
    seen = []

    def fake_post(url, *, headers, json, timeout):  # noqa: ANN001
        seen.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        content = """```json
{"objects":[{"label":"red cup","bbox_2d":[78,125,469,833],"point_2d":[234,417],"confidence":0.88}]}
```"""
        return DummyResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("real2sim_scene_foundry.proposals.requests.post", fake_post)

    proposals = QwenProposalClient(
        base_url="http://qwen.example/v1",
        api_key="secret",
        model="qwen-vl",
        timeout_s=3.0,
    ).propose(image, target_labels=["cup"])

    assert proposals[0].label == "red cup"
    assert proposals[0].bbox_xyxy == (5, 6, 30, 40)
    assert proposals[0].point_xy == (15, 20)
    assert proposals[0].source == "qwen"
    assert seen[0]["url"] == "http://qwen.example/v1/chat/completions"
    assert seen[0]["headers"]["Authorization"] == "Bearer secret"
    content = seen[0]["json"]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded)


def test_qwen_default_prompt_asks_for_all_tabletop_foreground_objects(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), color=(1, 2, 3)).save(image)
    seen = []

    def fake_post(url, *, headers, json, timeout):  # noqa: ANN001
        seen.append(json["messages"][0]["content"][0]["text"])
        return DummyResponse({"choices": [{"message": {"content": '{"objects":[{"label":"tissue pack"}]}'}}]})

    monkeypatch.setattr("real2sim_scene_foundry.proposals.requests.post", fake_post)

    QwenProposalClient(base_url="http://qwen.example/v1", api_key="secret").propose(image)

    assert "every visible movable foreground object" in seen[0]
    assert "tissue packs" in seen[0]
    assert "cloths" in seen[0]
    assert "Exclude the table surface" in seen[0]


def test_qwen_proposal_client_uses_qwen_proxy_env_and_scales_1000_coordinates(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (200, 100), color=(1, 2, 3)).save(image)
    monkeypatch.setenv("QWEN_PROXY_URL", "http://proxy.local/v1")
    monkeypatch.setenv("QWEN_PROXY_KEY", "proxy-secret")
    seen = []

    def fake_post(url, *, headers, json, timeout):  # noqa: ANN001
        seen.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"objects":[{"label":"cup","box_xyxy":[100,200,500,800],"point_xy":[300,500]}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("real2sim_scene_foundry.proposals.requests.post", fake_post)

    proposals = QwenProposalClient().propose(image, target_labels=["cup"])

    assert proposals[0].bbox_xyxy == (20, 20, 100, 80)
    assert proposals[0].point_xy == (60, 50)
    assert seen[0]["url"] == "http://proxy.local/v1/chat/completions"
    assert seen[0]["headers"]["Authorization"] == "Bearer proxy-secret"
    assert seen[0]["json"]["backend"] == "dashscope"


def test_qwen_proposal_client_scales_1000_coordinates_for_wide_images(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (1280, 720), color=(1, 2, 3)).save(image)

    def fake_post(url, *, headers, json, timeout):  # noqa: ANN001
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"objects":[{"label":"cup","bbox_2d":[659,380,774,551],"point_2d":[706,460]}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("real2sim_scene_foundry.proposals.requests.post", fake_post)

    proposals = QwenProposalClient(
        base_url="http://qwen.example/v1",
        api_key="secret",
        model="qwen-vl",
    ).propose(image, target_labels=["cup"])

    assert proposals[0].bbox_xyxy == (844, 274, 991, 397)
    assert proposals[0].point_xy == (904, 331)
