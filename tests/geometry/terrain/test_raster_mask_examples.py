import math

from nstk.geometry.raster_mask import RasterMask


def test_raster_mask_boolean_example_smoke():
    a = RasterMask(tolerance_deg=0.01)
    a.add_cap_azel(0.0, 0.0, 45.0)
    a.compile()

    b = RasterMask(tolerance_deg=0.01)
    b.add_cap_azel(20.0, 0.0, 35.0)
    b.compile()

    crescent = a - b
    added = a + b
    not_added = ~added

    assert a.node_count() > 0
    assert b.node_count() > 0
    assert crescent.node_count() > 0
    assert added.node_count() > 0
    assert not_added.node_count() > 0

    assert crescent.memory_bytes(include_python_overhead=True) > 0
    assert added.memory_bytes(include_python_overhead=True) > 0
    assert not_added.memory_bytes(include_python_overhead=True) > 0


def test_raster_mask_polygon_and_large_cap_example_smoke():
    polygon_fov = RasterMask.from_azel_polygon(
        vertices_az_el_deg=[(-60, -30), (60, -30), (60, 30), (-60, 30)],
        az_res=720,
        el_res=360,
        face_res=512,
        tolerance_deg=0.01,
    )

    large_cap_fov = RasterMask(tolerance_deg=0.05)
    large_cap_fov.add_cap_solid_angle_azel(0.0, 90.0, 2.0 * math.pi)
    large_cap_fov.compile()

    clipped = large_cap_fov & polygon_fov

    assert polygon_fov.node_count() > 0
    assert large_cap_fov.node_count() > 0
    assert clipped.node_count() > 0
    assert clipped.memory_bytes(include_python_overhead=True) > 0
