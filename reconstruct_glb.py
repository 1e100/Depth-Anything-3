#!/usr/bin/env python3
"""
Reconstruct a surface mesh from a GLB point cloud or mesh and save as PLY/OBJ.

Pipeline:
  1. Load GLB:
     * First try trimesh (for mesh / triangle GLBs).
     * If trimesh reports 0 vertices, fall back to pygltflib to read
       point cloud attributes (POSITION + COLOR_0).
  2. Get a colored point cloud (possibly downsampled).
  3. Estimate normals.
  4. Reconstruct surface (Poisson or Ball Pivoting) using Open3D.
  5. Transfer point colors to mesh vertices.
  6. Save mesh (PLY recommended for CloudCompare / MeshLab).

Dependencies:
  pip install trimesh[easy] open3d numpy pygltflib
"""

import argparse
import pathlib
import sys

import numpy as np
import open3d as o3d
import trimesh
from pygltflib import GLTF2


# ----------------------- glTF / GLB helpers -----------------------


def _get_accessor_array(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    """Return accessor data as a numpy array of shape [count, num_components].

    Supports both tightly packed and interleaved (byteStride > element_size)
    buffer views.
    """
    # Step: look up accessor, buffer view, and binary blob.
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()

    # Step: map glTF componentType to numpy dtype.
    component_type_to_dtype = {
        5120: np.int8,  # BYTE
        5121: np.uint8,  # UNSIGNED_BYTE
        5122: np.int16,  # SHORT
        5123: np.uint16,  # UNSIGNED_SHORT
        5125: np.uint32,  # UNSIGNED_INT
        5126: np.float32,  # FLOAT
    }
    if accessor.componentType not in component_type_to_dtype:
        raise RuntimeError(f"Unsupported componentType: {accessor.componentType}")

    dtype = component_type_to_dtype[accessor.componentType]

    # Step: map accessor.type to number of components.
    type_to_components = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    if accessor.type not in type_to_components:
        raise RuntimeError(f"Unsupported accessor type: {accessor.type}")

    num_components = type_to_components[accessor.type]

    # Step: compute offsets and stride.
    buffer_offset = buffer_view.byteOffset or 0
    accessor_offset = accessor.byteOffset or 0
    offset_base = buffer_offset + accessor_offset

    element_nbytes = np.dtype(dtype).itemsize * num_components
    stride = buffer_view.byteStride or element_nbytes

    count = accessor.count

    # Step: tightly packed case (most common).
    if stride == element_nbytes:
        raw = np.frombuffer(
            blob,
            dtype=dtype,
            count=count * num_components,
            offset=offset_base,
        )
        return raw.reshape((count, num_components))

    # Step: interleaved data (byteStride > element_nbytes).
    out = np.empty((count, num_components), dtype=dtype)
    for i in range(count):
        o = offset_base + i * stride
        out[i, :] = np.frombuffer(
            blob,
            dtype=dtype,
            count=num_components,
            offset=o,
        )
    return out


def _load_points_from_gltf(glb_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load point positions and optional colors from a GLB via pygltflib.

    Collects POSITION (and COLOR_0 if present) across all mesh primitives
    and concatenates them into one big point cloud.
    """
    # Step: load GLB structure + binary.
    gltf = GLTF2().load(str(glb_path))

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray | None] = []

    meshes = gltf.meshes or []
    for mesh_index, mesh in enumerate(meshes):
        for prim_index, prim in enumerate(mesh.primitives or []):
            attrs = prim.attributes  # This is a pygltflib.Attributes object.

            # Step: POSITION is a field, not a dict key.
            pos_idx = getattr(attrs, "POSITION", None)
            if pos_idx is None:
                continue

            # Step: read positions.
            pos_arr = _get_accessor_array(gltf, pos_idx)
            if pos_arr.shape[1] < 3:
                continue
            points = pos_arr[:, :3].astype(np.float32)

            # Step: read colors if COLOR_0 is present.
            colors = None
            col_idx = getattr(attrs, "COLOR_0", None)
            if col_idx is not None:
                col_accessor = gltf.accessors[col_idx]
                col_arr = _get_accessor_array(gltf, col_idx)

                if col_arr.shape[1] >= 3:
                    colors = col_arr[:, :3].astype(np.float32)

                    # Integer + normalized → map to [0,1].
                    if col_accessor.componentType in (
                        5120,
                        5121,
                        5122,
                        5123,
                        5125,
                    ):
                        max_val_map = {
                            5120: 127.0,
                            5121: 255.0,
                            5122: 32767.0,
                            5123: 65535.0,
                            5125: 4294967295.0,
                        }
                        max_val = max_val_map[col_accessor.componentType]
                        colors = colors / max_val
                    else:
                        # FLOAT colors – clamp just in case.
                        colors = np.clip(colors, 0.0, 1.0)

            all_points.append(points)
            all_colors.append(colors)

    # Step: ensure we actually found something.
    if not all_points:
        raise RuntimeError(
            "pygltflib fallback: no POSITION attributes found in GLB; "
            "cannot construct a point cloud."
        )

    points_concat = np.concatenate(all_points, axis=0)

    # Step: build colors if any primitive had colors.
    if any(c is not None for c in all_colors):
        color_chunks: list[np.ndarray] = []
        for pts, col in zip(all_points, all_colors):
            if col is None:
                color_chunks.append(np.full((pts.shape[0], 3), 0.7, dtype=np.float32))
            else:
                color_chunks.append(col[:, :3])
        colors_concat = np.concatenate(color_chunks, axis=0)
    else:
        colors_concat = None

    print(
        f"[DEBUG] pygltflib: loaded {points_concat.shape[0]} points "
        f"(colors={'yes' if colors_concat is not None else 'no'})",
        flush=True,
    )

    return points_concat, colors_concat


# ----------------------- point cloud construction -----------------------


def load_colored_point_cloud_from_glb(
    glb_path: pathlib.Path,
    sample_count: int,
) -> o3d.geometry.PointCloud:
    """Load a colored point cloud from a GLB using trimesh, then pygltflib fallback."""
    # Step: try loading via trimesh first (for normal mesh GLBs).
    loaded = trimesh.load(glb_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        print(
            f"[DEBUG] trimesh.Scene with {len(loaded.geometry)} geometries",
            flush=True,
        )
        mesh = loaded.to_mesh()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        mesh = None

    points: np.ndarray | None = None
    colors: np.ndarray | None = None

    # Step: if trimesh produced a valid mesh with vertices, use it.
    if mesh is not None and mesh.vertices is not None and len(mesh.vertices) > 0:
        v_count = len(mesh.vertices)
        f_count = len(mesh.faces) if mesh.faces is not None else 0
        print(f"[DEBUG] trimesh: vertices={v_count}, faces={f_count}", flush=True)

        # Step: if visuals are textured, bake them to per-vertex colors.
        if mesh.visual is not None and mesh.visual.kind == "texture":
            mesh.visual = mesh.visual.to_color()

        # Step: get per-vertex colors if available.
        vertex_colors = None
        if (
            hasattr(mesh, "visual")
            and mesh.visual is not None
            and hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
        ):
            vc = np.asarray(mesh.visual.vertex_colors)
            if vc.shape[0] == v_count:
                vertex_colors = vc[:, :3].astype(np.float32)
                if vertex_colors.max() > 1.0:
                    vertex_colors = vertex_colors / 255.0

        if f_count > 0:
            # --- Case 1: triangle mesh – sample uniformly on surface. ---
            samples, face_index, sample_colors = trimesh.sample.sample_surface(
                mesh, sample_count, sample_color=True
            )
            points = np.asarray(samples, dtype=np.float32)

            if sample_colors is not None and sample_colors.size > 0:
                c = np.asarray(sample_colors, dtype=np.float32)
                c = c[:, :3]
                if c.max() > 1.0:
                    c = c / 255.0
                colors = c
            elif vertex_colors is not None:
                colors = vertex_colors
        else:
            # --- Case 2: no faces – treat vertices as a point cloud. ---
            points = np.asarray(mesh.vertices, dtype=np.float32)
            colors = vertex_colors

    # Step: if trimesh produced no usable vertices, fall back to pygltflib.
    if points is None or points.size == 0:
        print(
            "[DEBUG] trimesh reports 0 vertices, falling back to pygltflib.",
            flush=True,
        )
        points, colors = _load_points_from_gltf(glb_path)

    # Step: downsample if point cloud is larger than requested sample_count.
    if points.shape[0] > sample_count:
        idx = np.random.choice(points.shape[0], size=sample_count, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    # Step: create Open3D point cloud with colors.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None and colors.size > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    else:
        # Assign neutral grey if no colors.
        grey = np.full((points.shape[0], 3), 0.7, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(grey)

    return pcd


# ----------------------- reconstruction pipeline -----------------------


def estimate_normals_for_point_cloud(pcd: o3d.geometry.PointCloud) -> None:
    """Estimate normals for an Open3D point cloud in-place."""
    # Step: choose radius based on bounding box size.
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    diag = float(np.linalg.norm(max_bound - min_bound))
    radius = max(diag * 0.01, 1e-6)

    # Step: estimate normals using hybrid KD-tree search.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    # Step: try to orient normals consistently.
    try:
        pcd.orient_normals_consistent_tangent_plane(10)
    except RuntimeError:
        # Orientation can fail on degenerate point clouds; normals are still usable.
        pass


def reconstruct_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int,
    density_filter: bool,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct a surface mesh using Poisson reconstruction."""
    # Step: run Poisson surface reconstruction.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Step: optionally filter low-density vertices to remove spurious surface.
    if density_filter:
        densities_np = np.asarray(densities, dtype=np.float64)
        density_threshold = float(np.quantile(densities_np, 0.01))
        mask_remove = densities_np < density_threshold
        mesh.remove_vertices_by_mask(mask_remove)

    return mesh


def reconstruct_ball_pivoting(
    pcd: o3d.geometry.PointCloud,
    radius_factor: float,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct a surface mesh using Ball Pivoting."""
    # Step: compute a base radius from point cloud bounding box.
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    diag = float(np.linalg.norm(max_bound - min_bound))
    base_radius = max(diag * radius_factor, 1e-6)

    # Step: configure radii for multi-scale BPA.
    radii = [base_radius, base_radius * 2.0, base_radius * 4.0]
    radii_vec = o3d.utility.DoubleVector(radii)

    # Step: run Ball Pivoting.
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_vec)

    return mesh


def transfer_vertex_colors_from_points(
    pcd: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
) -> None:
    """Transfer colors from a colored point cloud to mesh vertices."""
    # Step: exit early if there are no point colors.
    if not pcd.has_colors():
        return

    # Step: build KD-tree for nearest-neighbor color lookup.
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    pc_colors = np.asarray(pcd.colors, dtype=np.float64)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)

    # Step: allocate color array for all mesh vertices.
    new_colors = np.zeros_like(mesh_vertices, dtype=np.float64)

    # Step: for each mesh vertex, average colors of its k nearest points.
    k = 3
    for i, v in enumerate(mesh_vertices):
        _, idx, _ = kdtree.search_knn_vector_3d(v, k)
        neighbor_colors = pc_colors[idx, :]
        new_colors[i, :] = neighbor_colors.mean(axis=0)

    # Step: assign computed colors to the mesh.
    mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: pathlib.Path) -> None:
    """Save mesh to disk using Open3D (PLY recommended)."""
    # Step: default to .ply if no extension provided.
    if path.suffix == "":
        path = path.with_suffix(".ply")

    # Step: write mesh.
    ok = o3d.io.write_triangle_mesh(str(path), mesh)
    if not ok:
        raise RuntimeError(f"Failed to write mesh to {path}")


# ----------------------- CLI wrapper -----------------------


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a surface mesh from a GLB point cloud / mesh.\n"
            "Texture / color is preserved as vertex colors in the output."
        )
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input GLB file.",
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Output mesh path (.ply, .obj, etc). Defaults to <input>_recon.ply.",
    )

    parser.add_argument(
        "--method",
        choices=["poisson", "bpa"],
        default="poisson",
        help="Reconstruction method: poisson (default) or bpa (ball pivoting).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=150000,
        help="Maximum number of points to use for reconstruction.",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=10,
        help="Octree depth for Poisson reconstruction.",
    )
    parser.add_argument(
        "--bpa-radius-factor",
        type=float,
        default=0.01,
        help="Ball radius as a fraction of bounding-box diagonal (BPA only).",
    )
    parser.add_argument(
        "--no-density-filter",
        action="store_true",
        help="Disable density-based vertex trimming after Poisson.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Main CLI entry point."""
    # Step: parse CLI arguments and normalize paths.
    args = parse_args(argv)
    input_path = pathlib.Path(args.input).expanduser().resolve()

    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_recon.ply")
    else:
        output_path = pathlib.Path(args.output).expanduser().resolve()

    # Step: load colored point cloud (trimesh first, pygltflib fallback).
    pcd = load_colored_point_cloud_from_glb(input_path, args.samples)

    # Step: estimate normals.
    estimate_normals_for_point_cloud(pcd)

    # Step: reconstruct surface via chosen method.
    if args.method == "poisson":
        mesh = reconstruct_poisson(
            pcd,
            depth=args.poisson_depth,
            density_filter=not args.no_density_filter,
        )
    else:
        mesh = reconstruct_ball_pivoting(
            pcd,
            radius_factor=args.bpa_radius_factor,
        )

    # Step: transfer colors from points to mesh vertices.
    transfer_vertex_colors_from_points(pcd, mesh)

    # Step: save result.
    save_mesh(mesh, output_path)
    print(f"Saved reconstructed mesh to: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
