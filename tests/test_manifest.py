import pytest

from real2sim_scene_foundry.manifest import SceneBackground, SceneManifest, SceneObject


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


def test_manifest_serializes_background_branch():
    background = SceneBackground(
        source_backend="opencv_inpaint_single_frame",
        bg_only_image_path="background/bg_only.png",
        foreground_mask_path="background/foreground_mask.png",
        point_cloud_path="background/bg_only_cloud.ply",
        status="proxy_from_single_stereo_pair",
    )

    data = SceneManifest(objects=[], background=background).to_dict()

    assert data["background"]["source_backend"] == "opencv_inpaint_single_frame"
    assert data["background"]["bg_only_image_path"] == "background/bg_only.png"


def test_manifest_serializes_separated_asset_roles():
    obj = SceneObject(
        object_id="cup",
        label="cup",
        mesh_path="objects/cup/mesh_aligned.glb",
        mask_path="objects/cup/mask.png",
        crop_path="objects/cup/crop.png",
        T_object_to_camera=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        T_object_to_world=[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]],
        scale_m=0.3,
        mass_kg=0.2,
        friction=0.8,
        confidence=0.7,
        visual_asset={
            "path": "objects/cup/visual.glb",
            "source": "sam3d_mesh_glb_base64",
            "status": "ready",
            "final_visual": True,
        },
        collision_asset={
            "path": "objects/cup/collision.glb",
            "source": "convex_hull_from_visual_mesh",
            "status": "ready",
        },
        debug_proxy={
            "path": "objects/cup/debug_bbox.glb",
            "source": "bbox_from_visual_bounds",
            "status": "ready",
        },
        physics={
            "path": "objects/cup/physics.json",
            "mass_kg": 0.2,
            "friction": 0.8,
            "restitution": 0.0,
            "collision_source": "convex_hull_from_visual_mesh",
        },
    )

    data = SceneManifest(objects=[obj]).to_dict()

    assert data["objects"][0]["visual_asset"]["path"] == "objects/cup/visual.glb"
    assert data["objects"][0]["collision_asset"]["path"] == "objects/cup/collision.glb"
    assert data["objects"][0]["debug_proxy"]["path"] == "objects/cup/debug_bbox.glb"
    assert data["objects"][0]["physics"]["collision_source"] == "convex_hull_from_visual_mesh"
