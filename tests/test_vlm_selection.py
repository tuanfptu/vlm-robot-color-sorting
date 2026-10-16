"""
Unit tests for language-conditioned object selection.

Mocked tests run without CLIP / PyBullet. The real-CLIP test at the bottom runs
only when torch + transformers + pybullet are installed and the
openai/clip-vit-base-patch32 checkpoint can be loaded; otherwise it is skipped.

    python3 -m pytest tests/test_vlm_selection.py -v
"""

import importlib
import math
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import vlm_helpers as vh  # noqa: E402

CATALOG = ["red_cube", "red_sphere", "green_sphere", "yellow_cylinder", "blue_cube"]
ZONES = {
    "red_zone": {"position": [-0.1, 0.3, 0.65], "size": [0.15, 0.15, 0.02]},
    "blue_zone": {"position": [-0.1, -0.3, 0.65], "size": [0.15, 0.15, 0.02]},
    "green_zone": {"position": [0.7, 0.3, 0.65], "size": [0.15, 0.15, 0.02]},
    "yellow_zone": {"position": [0.7, -0.3, 0.65], "size": [0.15, 0.15, 0.02]},
}


class FakeImg:
    def __init__(self, size=(40, 40)):
        self.size = size


def crops_for(names):
    return {n: FakeImg() for n in names}


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd,expected", [
    ("Pick up the red cube", "red_cube"),
    ("Pick up the blue cube", "blue_cube"),
    ("Pick up the red sphere", "red_sphere"),
    ("Pick up the green sphere", "green_sphere"),
    ("Pick up the yellow cylinder", "yellow_cylinder"),
    ("grab the green ball", "green_sphere"),
    ("yellow can", "yellow_cylinder"),
    ("PICK UP THE BLUE BLOCK!", "blue_cube"),
])
def test_supported_commands_resolve_to_one_object(cmd, expected):
    parsed = vh.parse_command(cmd, CATALOG)
    assert parsed.status == vh.OK
    assert parsed.candidates == [expected]
    assert parsed.zone is None


def test_clip_text_is_the_object_phrase():
    parsed = vh.parse_command("Pick up the red cube", CATALOG)
    assert parsed.clip_text == "a red cube"


@pytest.mark.parametrize("cmd", ["", "   ", None, "clip:", "!!!"])
def test_empty_prompt(cmd):
    assert vh.parse_command(cmd, CATALOG).status == vh.EMPTY


@pytest.mark.parametrize("cmd", ["Pick up the purple cube", "pick the orange ball",
                                 "pick up the red pyramid", "grab the cone"])
def test_unsupported_color_or_shape(cmd):
    parsed = vh.parse_command(cmd, CATALOG)
    assert parsed.status == vh.UNSUPPORTED
    assert parsed.candidates == []


@pytest.mark.parametrize("cmd", ["Pick up the blue sphere", "pick up the green cube",
                                 "yellow ball"])
def test_object_not_in_simulator(cmd):
    parsed = vh.parse_command(cmd, CATALOG)
    assert parsed.status == vh.NOT_PRESENT
    assert parsed.candidates == []


@pytest.mark.parametrize("cmd,cands", [
    ("pick up the red object", {"red_cube", "red_sphere"}),
    ("pick up the cube", {"red_cube", "blue_cube"}),
    ("pick up the ball", {"red_sphere", "green_sphere"}),
    ("pick up the red or blue cube", {"red_cube", "blue_cube"}),
])
def test_ambiguous_commands(cmd, cands):
    parsed = vh.parse_command(cmd, CATALOG)
    assert parsed.status == vh.AMBIGUOUS
    assert set(parsed.candidates) == cands


def test_ungrounded_command_is_rejected_not_guessed():
    parsed = vh.parse_command("pick up the banana", CATALOG)
    assert parsed.status == vh.UNGROUNDED
    assert parsed.candidates == []


def test_freeform_prefix_keeps_original_clip_behaviour():
    parsed = vh.parse_command("clip: something round", CATALOG)
    assert parsed.status == vh.OK and parsed.freeform
    assert parsed.clip_text == "something round"
    assert set(parsed.candidates) == set(CATALOG)


@pytest.mark.parametrize("cmd,zone", [
    ("Pick up the red cube and put it in the blue zone", "blue_zone"),
    ("move the green sphere into the yellow area", "yellow_zone"),
    ("pick up the yellow cylinder and throw it away", "throw"),
])
def test_zone_in_command(cmd, zone):
    parsed = vh.parse_command(cmd, CATALOG)
    assert parsed.status == vh.OK
    assert parsed.zone == zone
    assert len(parsed.candidates) == 1


@pytest.mark.parametrize("cmd", ["put the red cube in the purple zone",
                                 "red cube to zone 7"])
def test_invalid_zone_in_command(cmd):
    assert vh.parse_command(cmd, CATALOG).status == vh.INVALID_ZONE


def test_zone_color_does_not_leak_into_object_color():
    parsed = vh.parse_command("put the red cube in the blue zone", CATALOG)
    assert parsed.color == "red" and parsed.candidates == ["red_cube"]


# ---------------------------------------------------------------------------
# Score ranking and selection
# ---------------------------------------------------------------------------

def test_rank_scores_sorts_and_drops_invalid():
    ranking, invalid = vh.rank_scores({
        "red_cube": 0.2, "blue_cube": 0.5, "bad_nan": float("nan"),
        "bad_inf": float("inf"), "bad_str": "x", "bad_none": None, "green_sphere": 0.3,
    })
    assert [n for n, _ in ranking] == ["blue_cube", "green_sphere", "red_cube"]
    assert set(invalid) == {"bad_nan", "bad_inf", "bad_str", "bad_none"}


def test_rank_scores_empty_and_ties_deterministic():
    assert vh.rank_scores({}) == ([], [])
    assert vh.rank_scores(None) == ([], [])
    ranking, _ = vh.rank_scores({"b": 0.5, "a": 0.5})
    assert [n for n, _ in ranking] == ["a", "b"]


SCORES = {"red_cube": 0.45, "red_sphere": 0.30, "blue_cube": 0.10,
          "green_sphere": 0.10, "yellow_cylinder": 0.05}


def test_select_red_cube():
    sel = vh.select_target(vh.parse_command("Pick up the red cube", CATALOG), SCORES)
    assert sel.status == vh.OK and sel.selected == "red_cube" and sel.clip_agrees


def test_select_blue_cube_even_if_clip_prefers_other_object():
    sel = vh.select_target(vh.parse_command("Pick up the blue cube", CATALOG), SCORES)
    assert sel.status == vh.OK and sel.selected == "blue_cube"
    assert sel.clip_agrees is False and sel.clip_top == "red_cube"


def test_select_green_sphere():
    scores = {"green_sphere": 0.6, "red_sphere": 0.2, "red_cube": 0.2}
    sel = vh.select_target(vh.parse_command("Pick up the green sphere", CATALOG), scores)
    assert sel.status == vh.OK and sel.selected == "green_sphere"


def test_ambiguous_selection_does_not_auto_pick():
    sel = vh.select_target(vh.parse_command("pick up the red object", CATALOG), SCORES)
    assert sel.status == vh.AMBIGUOUS
    assert sel.selected is None
    assert [n for n, _ in sel.candidate_ranking] == ["red_cube", "red_sphere"]


def test_target_not_visible():
    scores = {"red_cube": 0.7, "blue_cube": 0.3}  # green sphere not in this frame
    sel = vh.select_target(vh.parse_command("Pick up the green sphere", CATALOG), scores)
    assert sel.status == vh.NOT_VISIBLE and sel.selected is None


def test_no_scores():
    sel = vh.select_target(vh.parse_command("Pick up the red cube", CATALOG), {})
    assert sel.status == vh.NO_SCORES and sel.selected is None
    sel = vh.select_target(vh.parse_command("Pick up the red cube", CATALOG),
                           {"red_cube": float("nan")})
    assert sel.status == vh.NO_SCORES


def test_validate_crops_handles_missing_and_bad_crops():
    bboxes = {"red_cube": (0, 0, 10, 10), "blue_cube": (0, 0, 10, 10)}
    crops = {"red_cube": FakeImg(), "ghost": FakeImg(), "green_sphere": None,
             "yellow_cylinder": FakeImg((1, 30))}
    valid, problems = vh.validate_crops(crops, bboxes, CATALOG)
    assert list(valid) == ["red_cube"]
    joined = " ".join(problems)
    assert "blue_cube" in joined and "ghost" in joined
    assert "green_sphere" in joined and "yellow_cylinder" in joined
    assert vh.validate_crops({}, {}, CATALOG) == ({}, [])


# ---------------------------------------------------------------------------
# Zones and verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,dest", [
    ("1", "red_zone"), ("2", "blue_zone"), ("3", "green_zone"), ("4", "yellow_zone"),
    ("5", "throw"), ("s", "safe"), ("Blue", "blue_zone"), ("yellow zone", "yellow_zone"),
    ("throw it away", "throw"),
])
def test_zone_choice_valid(text, dest):
    assert vh.parse_zone_choice(text, ZONES) == dest


@pytest.mark.parametrize("text", ["", "9", "0", "purple", "abc", None])
def test_zone_choice_invalid(text):
    assert vh.parse_zone_choice(text, ZONES) is None


def test_verify_in_zone():
    z = ZONES["red_zone"]
    assert vh.verify_in_zone((-0.1, 0.3, 0.68), z)
    assert vh.verify_in_zone((-0.04, 0.36, 0.68), z)       # near edge, within tolerance
    assert not vh.verify_in_zone((0.2, 0.3, 0.68), z)       # wrong x
    assert not vh.verify_in_zone((-0.1, 0.3, 0.02), z)      # fell to the floor
    assert not vh.verify_in_zone(None, z)


def test_is_lifted():
    assert vh.is_lifted((0.6, 0.0, 0.65), (0.6, 0.0, 0.90))
    assert not vh.is_lifted((0.6, 0.0, 0.65), (0.6, 0.0, 0.66))
    assert not vh.is_lifted(None, (0, 0, 1))


def test_score_table_marks_selection():
    vh.Style.enabled = False
    lines = vh.score_table([("red_cube", 0.6), ("blue_cube", 0.4)], "red_cube", ["red_cube"])
    assert "★ red_cube" in lines[1] and "· blue_cube" in lines[2]
    assert vh.score_table([], None) == ["(no scores)"]


# ---------------------------------------------------------------------------
# End-to-end handle_command() with a mocked simulator (no CLIP / PyBullet)
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_module(monkeypatch):
    """Import color_sorting_vlm with heavy dependencies stubbed out."""
    positions = {}

    pb = types.ModuleType("pybullet")
    pb.getBasePositionAndOrientation = lambda oid: (positions[oid], (0, 0, 0, 1))
    pb.stepSimulation = lambda: None
    pvs = types.ModuleType("panda_vision_simulation")
    pvs.VisionLanguagePandaSimulation = object
    mpl = types.ModuleType("matplotlib")
    plt = types.ModuleType("matplotlib.pyplot")
    mpl.pyplot = plt
    for name, mod in {"pybullet": pb, "panda_vision_simulation": pvs, "torch": types.ModuleType("torch"),
                      "matplotlib": mpl, "matplotlib.pyplot": plt}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "color_sorting_vlm", raising=False)
    mod = importlib.import_module("color_sorting_vlm")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._positions = positions
    return mod


class FakeSim:
    """Mimics the parts of VisionLanguagePandaSimulation used by handle_command."""

    def __init__(self, positions, visible=CATALOG, scores=None, lift=True, raise_single=False):
        self.objects = {n: {"id": i, "config": {"size": 0.05}} for i, n in enumerate(CATALOG)}
        self.sorting_zones = ZONES
        self.place_position = [0.2, 0.3, 0.6]
        self.home_position = [0.3, 0.0, 0.85]
        self.gripper_open_position = 0.04
        self.visible = list(visible)
        self.scores = scores
        self.lift = lift
        self.raise_single = raise_single
        self.positions = positions
        self.held = None
        self.calls = []
        for n, info in self.objects.items():
            positions[info["id"]] = (0.5, 0.0, 0.65)

    def capture_and_process_scene(self):
        bboxes = {n: (0, 0, 10, 10) for n in self.visible}
        return "IMG", bboxes, crops_for(self.visible)

    def compute_object_similarity(self, crops, text):
        self.calls.append(("clip", text, sorted(crops)))
        if self.raise_single and len(crops) == 1:
            raise IndexError("invalid index of a 0-dim tensor")
        if self.scores is not None:
            return {n: s for n, s in self.scores.items() if n in crops}
        words = text.lower().split()
        raw = {n: 1.0 + sum(p in words for p in n.split("_")) * 2.0 for n in crops}
        z = sum(math.exp(v) for v in raw.values())
        return {n: math.exp(v) / z for n, v in raw.items()}

    def pick_object(self, name):
        self.calls.append(("pick", name))
        if name not in self.objects:
            return False
        oid = self.objects[name]["id"]
        if self.lift:
            x, y, z = self.positions[oid]
            self.positions[oid] = (x, y, z + 0.3)
            self.held = name
        return True

    # Used by the original placement helpers in color_sorting_vlm.py
    def safe_move_to_position(self, pos):
        self.move_to_position(pos)

    def move_to_position(self, pos):
        self.calls.append(("move", tuple(pos)))
        self._last = pos

    def control_gripper(self, value):
        self.calls.append(("gripper", value))
        if value == self.gripper_open_position and self.held:
            oid = self.objects[self.held]["id"]
            x, y, _ = self._last
            self.positions[oid] = (x, y, 0.68 if x > -0.25 else 0.02)
            self.held = None


def answers(*seq):
    it = iter(seq)
    return lambda prompt="": next(it)


def picks(sim):
    return [c for c in sim.calls if c[0] == "pick"]


def test_e2e_red_cube_to_red_zone_verified(csv_module):
    sim = FakeSim(csv_module._positions)
    res = csv_module.handle_command(sim, "Pick up the red cube", CATALOG, ask=answers("y", "1"))
    assert res["selected"] == "red_cube"
    assert res["destination"] == "red_zone"
    assert res["placement"] == "VERIFIED"
    assert ("clip", "a red cube", sorted(CATALOG)) in sim.calls


def test_e2e_zone_from_command(csv_module):
    sim = FakeSim(csv_module._positions)
    res = csv_module.handle_command(sim, "pick up the blue cube and put it in the green zone",
                                    CATALOG, ask=answers("y"))
    assert res["selected"] == "blue_cube" and res["destination"] == "green_zone"
    assert res["placement"] == "VERIFIED"


def test_e2e_invalid_prompt_never_moves_robot(csv_module):
    for prompt in ["", "pick up the purple cube", "pick up the blue sphere", "banana",
                   "red cube to the pink zone"]:
        sim = FakeSim(csv_module._positions)
        res = csv_module.handle_command(sim, prompt, CATALOG, ask=answers())
        assert res["selected"] is None, prompt
        assert picks(sim) == [], prompt


def test_e2e_no_detected_objects(csv_module):
    sim = FakeSim(csv_module._positions, visible=[])
    res = csv_module.handle_command(sim, "Pick up the red cube", CATALOG, ask=answers())
    assert res["status"] == vh.NO_OBJECTS and picks(sim) == []


def test_e2e_target_not_visible(csv_module):
    sim = FakeSim(csv_module._positions, visible=["red_cube", "blue_cube"])
    res = csv_module.handle_command(sim, "Pick up the green sphere", CATALOG, ask=answers())
    assert res["status"] == vh.NOT_VISIBLE and picks(sim) == []


def test_e2e_ambiguous_requires_choice(csv_module):
    sim = FakeSim(csv_module._positions)
    res = csv_module.handle_command(sim, "pick up the red object", CATALOG, ask=answers(""))
    assert res["status"] == "CANCELLED" and picks(sim) == []
    sim = FakeSim(csv_module._positions)
    res = csv_module.handle_command(sim, "pick up the red object", CATALOG, ask=answers("2", "y", "1"))
    assert res["selected"] in ("red_cube", "red_sphere") and res["placement"] == "VERIFIED"


def test_e2e_invalid_zone_input_reprompts_then_safe(csv_module):
    sim = FakeSim(csv_module._positions)
    res = csv_module.handle_command(sim, "Pick up the yellow cylinder", CATALOG,
                                    ask=answers("y", "9", "purple", "x"))
    assert res["destination"] == "safe"
    assert res["placement"] == "VERIFIED"


def test_e2e_grasp_failure_is_reported(csv_module):
    sim = FakeSim(csv_module._positions, lift=False)
    res = csv_module.handle_command(sim, "Pick up the red cube", CATALOG, ask=answers("y"))
    assert res["status"] == "PICK_FAILED" and res["placement"] is None


def test_e2e_placement_outside_zone_is_not_reported_as_success(csv_module, monkeypatch):
    def drop_short(s, name, zone):  # controller "returns" but releases at the wrong spot
        s.move_to_position([0.3, 0.0, 0.7])
        s.control_gripper(s.gripper_open_position)

    monkeypatch.setattr(csv_module, "place_object_in_specific_zone", drop_short)
    sim = FakeSim(csv_module._positions)
    res = csv_module.handle_command(sim, "Pick up the red cube", CATALOG, ask=answers("y", "1"))
    assert res["placement"] == "NOT_IN_ZONE"


def test_e2e_single_visible_crop_workaround(csv_module):
    sim = FakeSim(csv_module._positions, visible=["red_cube"], raise_single=True)
    res = csv_module.handle_command(sim, "Pick up the red cube", CATALOG, ask=answers("y", "1"))
    assert res["selected"] == "red_cube" and res["placement"] == "VERIFIED"


# ---------------------------------------------------------------------------
# Optional: real CLIP inference on the real simulator (headless)
# ---------------------------------------------------------------------------

def _real_stack_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import pybullet  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(os.environ.get("RUN_REAL_CLIP") != "1" or not _real_stack_available(),
                    reason="set RUN_REAL_CLIP=1 with torch/transformers/pybullet + CLIP checkpoint")
def test_real_clip_selection_headless():
    for name in ("panda_vision_simulation", "color_sorting_vlm"):
        sys.modules.pop(name, None)
    from panda_vision_simulation import VisionLanguagePandaSimulation
    import pybullet as p

    sim = VisionLanguagePandaSimulation(gui_mode=False, model_name="openai/clip-vit-base-patch32")
    try:
        sim.setup_simulation()
        for _ in range(240):
            p.stepSimulation()
        sim.set_initial_robot_pose()
        catalog = list(sim.objects)
        _, bboxes, crops = sim.capture_and_process_scene()
        valid, _ = vh.validate_crops(crops, bboxes, catalog)
        assert valid, "no crops from real camera"
        for cmd, expected in [("Pick up the red cube", "red_cube"),
                              ("Pick up the blue cube", "blue_cube"),
                              ("Pick up the green sphere", "green_sphere"),
                              ("Pick up the yellow cylinder", "yellow_cylinder"),
                              ("Pick up the red sphere", "red_sphere")]:
            parsed = vh.parse_command(cmd, catalog)
            scores = sim.compute_object_similarity(valid, parsed.clip_text)
            sel = vh.select_target(parsed, scores)
            print(cmd, "->", sel.selected, "clip_top:", sel.clip_top, sel.ranking)
            assert sel.selected == expected
    finally:
        sim.cleanup()


# ---------------------------------------------------------------------------
# Visual control panel (mode 3) driven through its callbacks, headless
# ---------------------------------------------------------------------------

def test_visual_panel_flow(csv_module, monkeypatch):
    mpl = pytest.importorskip("matplotlib")
    for name in ("matplotlib", "matplotlib.pyplot"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as real_plt

    class Sim(FakeSim):
        def capture_camera_image(self):
            return None

    sim = Sim(csv_module._positions)
    gui = csv_module.VisualControlPanel(sim, real_plt)

    gui.text_box.set_val("pick up the purple cube")
    gui.on_analyze()
    gui.on_execute()                                   # nothing selected -> no robot motion
    assert gui.selected is None and picks(sim) == []

    gui.text_box.set_val("pick up the ball")
    gui.on_analyze()
    assert gui.selected is None                        # ambiguous: user must choose
    gui.on_candidate(0)
    assert gui.selected in ("red_sphere", "green_sphere")

    gui.text_box.set_val("Pick up the red cube")
    gui.on_analyze()
    assert gui.selected == "red_cube"
    gui.on_execute()                                   # no zone yet -> refuses
    assert picks(sim) == []
    gui.on_zone("red_zone")
    gui.on_execute()
    assert picks(sim) == [("pick", "red_cube")]
    real_plt.close(gui.fig)
