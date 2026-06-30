import pytest

from real2sim_scene_foundry.manifest import SceneManifest, SceneObject


def test_manifest_rejects_negative_mass():
    obj = SceneObject(
        object_id="cup",
        label="cup",
        mesh_path="objects/cup/mesh.glb",
        mask_path="objects/cup/mask.png",
        crop_path="objects/cup/crop.png",
        T_object_to_camera=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        T_object_to_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        scale_m=1.0,
        mass_kg=-0.1,
        friction=0.8,
        confidence=0.9,
    )

    with pytest.raises(ValueError, match="mass"):
        SceneManifest(objects=[obj]).validate()


def test_manifest_serializes_coordinate_conventions():
    obj = SceneObject(
        object_id="cup",
        label="cup",
        mesh_path="objects/cup/mesh.glb",
        mask_path="objects/cup/mask.png",
        crop_path="objects/cup/crop.png",
        T_object_to_camera=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        T_object_to_world=[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]],
        scale_m=0.3,
        mass_kg=0.2,
        friction=0.8,
        confidence=0.7,
    )

    data = SceneManifest(objects=[obj]).to_dict()

    assert data["coordinate_frames"]["camera"] == "opencv_x_right_y_down_z_forward_meters"
    assert data["coordinate_frames"]["world"] == "z_up_ground_plane_meters"
    assert data["objects"][0]["source_backend"] == "unknown"
