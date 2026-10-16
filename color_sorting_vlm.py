#!/usr/bin/env python3
"""
Vision-Language Color Sorting Demonstration

This script demonstrates automated color sorting using the Panda robot and a vision-language model (CLIP).
The robot identifies colored objects using CLIP and sorts them into matching colored zones.

This is a pretrained VLM (CLIP, zero-shot) used for object *selection*, combined
with a programmed robot controller. It is not an end-to-end trained VLA policy.

Pipeline (interactive mode):
    user text -> command validation (vlm_helpers.parse_command)
              -> camera image + segmentation crops (sim.capture_and_process_scene)
              -> CLIP similarity (sim.compute_object_similarity)
              -> selected object (vlm_helpers.select_target)
              -> sim.pick_object(name) -> placement zone -> verification in sim

Features:
- 5 colored objects (red cube, red sphere, blue cube, green sphere, yellow cylinder)
- 4 colored sorting zones (2 on left, 2 on right of robot)
- Vision-language (CLIP) based object identification and selection
- Automated color matching and placement
- Interactive text prompt mode with command validation
- Visual control panel (mode 3): camera, sorted CLIP scores, zone buttons
- Placement verified from the simulator state (not from the motion call returning)

Usage:
    python3 color_sorting_vlm.py
"""

from panda_vision_simulation import VisionLanguagePandaSimulation
import pybullet as p
import time
import matplotlib.pyplot as plt
import torch

from vlm_helpers import (
    AMBIGUOUS, OK, NO_OBJECTS, ZONE_MENU, SPECIAL_DESTINATIONS, FREEFORM_PREFIX, SHAPE_SYNONYMS,
    SUPPORTED_COLORS, Style, banner, panel, rule, score_table, paint_object,
    parse_command, validate_crops, select_target, parse_zone_choice,
    zone_side, verify_in_zone, is_lifted, describe_catalog, example_commands,
    split_object_name,
)

S = Style.s
MAX_ZONE_ATTEMPTS = 3


def main():
    """
    Run the color sorting demonstration.
    """
    print(banner("VLM Robot Color Sorting", "CLIP (openai/clip-vit-base-patch32) + Franka Panda"))
    print()
    print(S("  Choose your demo mode:", "bold"))
    print(f"   {S('1', 'bold', 'cyan')}  Automatic color sorting  {S('(sort all objects by color)', 'gray')}")
    example = S('(e.g. "Pick up the red cube")', 'gray')
    print(f"   {S('2', 'bold', 'cyan')}  Interactive text prompts {example}")
    print(f"   {S('3', 'bold', 'cyan')}  Visual control panel     {S('(window with camera, CLIP scores, buttons)', 'gray')}")
    print()

    choice = input(S("  Select mode (1/2/3, default=1): ", "bold")).strip()
    if choice not in ("", "1", "2", "3"):
        print(S(f"  Unknown mode '{choice}', using 1 (automatic).", "yellow"))
        choice = "1"

    # Create simulation (using CLIP model for fast processing)
    sim = VisionLanguagePandaSimulation(
        gui_mode=True,
        model_name="openai/clip-vit-base-patch32"
    )

    try:
        # Setup simulation environment with sorting zones
        print("Setting up simulation environment...")
        sim.setup_simulation()

        # Wait for physics to settle
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1./240.)

        # Set robot to initial pose
        sim.set_initial_robot_pose()

        print()
        print(scene_overview(sim))

        # Show initial scene
        print("\nShowing initial scene with objects and sorting zones...")
        initial_image = sim.capture_camera_image()

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(initial_image)
        ax.set_title("Initial Scene - Objects and Color Sorting Zones")
        ax.axis('off')

        # Zone annotations derived from the simulator configuration (robot frame)
        zone_annotations = [
            f"{zone.replace('_zone', '').upper()} ZONE\n({zone_side(cfg['position'])}, robot view)"
            for zone, cfg in sim.sorting_zones.items()
        ]

        # Add text annotations (approximate positions)
        ax.text(50, 50, "SORTING ZONES:", fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        for i, annotation in enumerate(zone_annotations):
            ax.text(50, 100 + i*40, f"• {annotation}", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))

        plt.tight_layout()
        plt.show()

        # Run selected demo mode
        if choice == "2":
            # Interactive text prompts only
            run_interactive_text_prompt_demo(sim)
        elif choice == "3":
            # Same pipeline as mode 2, in a single window
            run_visual_panel_demo(sim)
        else:
            # Automatic color sorting (default)
            print("\n🎨 Starting automatic color sorting...")
            results = sim.run_color_sorting_demo(visualize=True)
            report_sorting_results(sim, results)

        print("\nPress Enter to exit...")
        input()

    except KeyboardInterrupt:
        print("\nColor sorting interrupted by user.")
    except Exception as e:
        print(f"Error during color sorting: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.cleanup()


# ---------------------------------------------------------------------------
# Scene / verification helpers (read simulator state, never move the robot)
# ---------------------------------------------------------------------------

def scene_overview(sim):
    """Panel listing the objects and zones that actually exist in the simulator."""
    lines = [S("Objects:", "bold")]
    for name in sim.objects:
        lines.append(f"  • {paint_object(name)}")
    lines.append(S("Zones (robot frame: base at origin, facing +x):", "bold"))
    for zone, cfg in sim.sorting_zones.items():
        x, y, _ = cfg['position']
        lines.append(f"  • {paint_object(zone)}  {S(zone_side(cfg['position']), 'gray')}"
                     f"  {S(f'(x={x:+.2f}, y={y:+.2f})', 'gray')}")
    return panel("Scene", lines, kind="info")


def get_object_position(sim, object_name):
    """Ground-truth object position from PyBullet, or None if unavailable."""
    try:
        pos, _ = p.getBasePositionAndOrientation(sim.objects[object_name]['id'])
        return tuple(pos)
    except Exception:
        return None


def settle(steps=120):
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1./240.)


def verify_placement(sim, object_name, destination):
    """
    Check where the object really ended up.

    Returns (status, position) with status one of
    'VERIFIED', 'NOT_IN_ZONE', 'UNVERIFIED'.
    """
    pos = get_object_position(sim, object_name)
    if pos is None:
        return "UNVERIFIED", None
    if destination in sim.sorting_zones:
        ok = verify_in_zone(pos, sim.sorting_zones[destination])
    elif destination == "throw":
        ok = pos[2] < 0.4            # fell off the table onto the floor
    elif destination == "safe":
        tx, ty = sim.place_position[0], sim.place_position[1]
        ok = abs(pos[0] - tx) <= 0.1 and abs(pos[1] - ty) <= 0.1 and pos[2] >= 0.55
    else:
        return "UNVERIFIED", pos
    return ("VERIFIED" if ok else "NOT_IN_ZONE"), pos


def _fmt_pos(pos):
    return "unknown" if pos is None else f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f})"


def report_sorting_results(sim, results):
    """Auto-mode summary: controller-reported result vs. verified final position."""
    print()
    print(rule("═"))
    print(S("  COLOR SORTING RESULT (verified from simulator state)", "bold"))
    print(rule("═"))
    verified = 0
    for r in results:
        name = r['object_name']
        color, _ = split_object_name(name)
        target = f"{color}_zone" if color else None
        claimed = "moved" if r.get('success') else "failed"
        if target in sim.sorting_zones and r.get('success'):
            status, pos = verify_placement(sim, name, target)
        else:
            status, pos = ("NOT_IN_ZONE" if r.get('success') else "FAILED"), get_object_position(sim, name)
        verified += status == "VERIFIED"
        mark = {"VERIFIED": S("✔ VERIFIED", "green", "bold"),
                "NOT_IN_ZONE": S("✖ NOT IN ZONE", "red", "bold"),
                "FAILED": S("✖ FAILED", "red", "bold")}.get(status, S("? UNVERIFIED", "yellow"))
        label = paint_object(name) + " " * max(0, 16 - len(name))
        print(f"  {label} → {target or '?':<12} controller: {claimed:<7} {mark}  {S(_fmt_pos(pos), 'gray')}")
    print(rule())
    total = len(results)
    if total and verified == total:
        print(S(f"  🏆 {verified}/{total} objects verified inside their matching zone.", "green", "bold"))
    else:
        print(S(f"  ⚠️  {verified}/{total} objects verified inside their matching zone.", "yellow", "bold"))


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_help(catalog):
    lines = [
        S("Say what to pick, by color and/or shape:", "bold"),
        *[f"  {S('›', 'cyan')} {ex}" for ex in example_commands(catalog)],
        "",
        S("Optional placement in the same command:", "bold"),
        f"  {S('›', 'cyan')} Pick up the red cube and put it in the blue zone",
        f"  {S('›', 'cyan')} Pick up the yellow cylinder and throw it away",
        "",
        f"Colors: {', '.join(SUPPORTED_COLORS)}   Shapes: {', '.join(SHAPE_SYNONYMS)}"
        f" {S('(ball, box, block, can… also understood)', 'gray')}",
        f"Free-form CLIP (no validation): {S(FREEFORM_PREFIX + ' something round', 'cyan')}",
        f"Commands: {S('help', 'cyan')}, {S('quit', 'cyan')}",
    ]
    return panel("Interactive text prompts", lines, kind="info")


def run_interactive_text_prompt_demo(sim):
    """
    Run an interactive demo where the user can enter text prompts
    to select and manipulate objects.

    Args:
        sim: VisionLanguagePandaSimulation instance
    """
    catalog = list(sim.objects.keys())
    print()
    print(interactive_help(catalog))

    while True:
        try:
            # Get user input
            prompt = input(f"\n{S('🤖 Command', 'bold', 'cyan')} {S('›', 'cyan')} ").strip()

            # Check for exit commands
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting interactive demo...")
                break
            if prompt.lower() in ['help', '?', 'h']:
                print(interactive_help(catalog))
                continue

            result = handle_command(sim, prompt, catalog)

            # Optional: show images side by side
            if result.get('camera_image') is not None and result.get('picked'):
                print("\n📸 Updated scene:")
                final_image = sim.capture_camera_image()
                show_images = input("Show before/after images? (y/n): ").lower()
                if show_images in ['y', 'yes']:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    ax1.imshow(result['camera_image'])
                    ax1.set_title("Before")
                    ax1.axis('off')

                    ax2.imshow(final_image)
                    ax2.set_title("After")
                    ax2.axis('off')

                    plt.tight_layout()
                    plt.show()

        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("Interactive demo completed!")


def compute_clip_scores(sim, crops, text):
    """
    Call the existing CLIP ranking (sim.compute_object_similarity).

    With exactly one visible crop the upstream softmax/indexing can fail
    (0-dim tensor); in that case the only crop is returned with score 1.0 and a
    note, because a softmax over a single item is 1.0 by definition.
    """
    try:
        return sim.compute_object_similarity(crops, text), None
    except (IndexError, RuntimeError, TypeError) as e:
        if len(crops) == 1:
            name = next(iter(crops))
            return {name: 1.0}, f"single visible crop - softmax is trivially 1.0 ({type(e).__name__})"
        raise


def ask_destination(sim, ask=input):
    """Placement menu with validation. Returns a zone name, 'throw' or 'safe'."""
    print(f"📍 {S('Where should it go?', 'bold')}")
    labels = {"throw": "🗑️  Throw away (fast throw!)", "safe": "Safe default position"}
    for key, dest in ZONE_MENU:
        if dest in sim.sorting_zones:
            cfg = sim.sorting_zones[dest]
            print(f"   {S(key, 'bold', 'cyan')}. {paint_object(dest)}  {S(zone_side(cfg['position']), 'gray')}")
        else:
            print(f"   {S(key, 'bold', 'cyan')}. {labels[dest]}")
    for attempt in range(1, MAX_ZONE_ATTEMPTS + 1):
        dest = parse_zone_choice(ask(f"Choose destination (1-5, s): "), sim.sorting_zones)
        if dest:
            return dest
        left = MAX_ZONE_ATTEMPTS - attempt
        if left:
            print(S(f"   ✖ Invalid placement zone. {left} attempt(s) left.", "red"))
    print(S("   ✖ No valid zone given - using the SAFE default position "
            "(the object is held and must be released).", "yellow"))
    return "safe"


def execute_placement(sim, object_name, destination):
    if destination in sim.sorting_zones:
        place_object_in_specific_zone(sim, object_name, destination)
    elif destination == "throw":
        throw_object_away(sim, object_name)
    else:
        safe_place_object(sim, object_name)


def analyze_command(sim, prompt, catalog):
    """
    Language + vision part of the pipeline. Never moves the robot.

    Returns dict: status, parsed, selection, camera_image, bounding_boxes.
    status is OK / AMBIGUOUS or the rejection code.
    """
    out = {'status': None, 'parsed': None, 'selection': None,
           'camera_image': None, 'bounding_boxes': {}}

    # 1) Language: validate command against the objects that exist in the sim
    parsed = parse_command(prompt, catalog)
    out['parsed'] = parsed
    print(rule())
    print(f"🔍 {S('Command', 'bold')}: {S(repr(prompt), 'cyan')}")
    if parsed.status not in (OK, AMBIGUOUS):
        print(panel(f"Rejected: {parsed.status}", [parsed.message,
                    S("Type 'help' for supported commands. Nothing was moved.", "gray")], kind="error"))
        out['status'] = parsed.status
        return out
    detail = []
    if parsed.color or parsed.shape:
        detail.append(f"color={parsed.color or '-'}  shape={parsed.shape or '-'}")
    if parsed.zone:
        detail.append(f"zone={parsed.zone}")
    detail.append(f"CLIP text: {S(repr(parsed.clip_text), 'cyan')}")
    print(f"   {S(' · '.join(detail), 'gray')}")

    # 2) Vision: camera image -> segmentation crops
    print(f"📸 {S('Capturing scene with segmentation...', 'bold')}")
    camera_image, bounding_boxes, crops = sim.capture_and_process_scene()
    out['camera_image'], out['bounding_boxes'] = camera_image, bounding_boxes or {}
    valid_crops, problems = validate_crops(crops, bounding_boxes, catalog)
    for msg in problems:
        print(S(f"   ! {msg}", "yellow"))
    if not valid_crops:
        print(panel("No objects detected", ["The camera sees no usable object crops."], kind="error"))
        out['status'] = NO_OBJECTS
        return out

    # 3) CLIP similarity (original ranking mechanism)
    scores, note = compute_clip_scores(sim, valid_crops, parsed.clip_text)
    if note:
        print(S(f"   ! {note}", "yellow"))
    selection = select_target(parsed, scores)
    out['selection'] = selection

    print(f"\n📊 {S('CLIP similarity (sorted)', 'bold')}")
    for line in score_table(selection.ranking, selection.selected, parsed.candidates):
        print(line)
    print(S("   ★ selected  ◆ matches command  · other    "
            "scores are relative, not calibrated confidence", "gray"))

    if selection.status not in (OK, AMBIGUOUS):
        print(panel(f"Rejected: {selection.status}", [selection.message,
                    S("Nothing was moved.", "gray")], kind="error"))
    elif selection.status == OK and not selection.clip_agrees:
        print(S(f"   ! {selection.message}", "yellow"))
    out['status'] = selection.status
    return out


def pick_verified(sim, selected):
    """
    Pick with the existing controller, then check the object actually rose.
    On a failed grasp the gripper is opened and the robot returns home.

    Returns (picked_call_ok, grasp_ok).
    """
    print(f"🤖 Picking up {selected}...")
    before = get_object_position(sim, selected)
    if not sim.pick_object(selected):
        print(panel("Pick failed", [f"sim.pick_object('{selected}') returned False."], kind="error"))
        return False, False
    after = get_object_position(sim, selected)
    if before is not None and after is not None and not is_lifted(before, after):
        print(panel("Grasp not verified", [
            f"Object did not rise with the gripper: {_fmt_pos(before)} → {_fmt_pos(after)}",
            "Releasing and returning home; nothing will be placed."], kind="error"))
        sim.control_gripper(sim.gripper_open_position)
        sim.move_to_position(sim.home_position)
        return True, False
    return True, True


def place_held_object(sim, selected, destination):
    """
    Place the held object, then verify from simulator state (not from the
    motion call returning). Returns (placement_status, position).
    """
    print(f"🚚 Moving {paint_object(selected)} → {S(destination, 'bold')}")
    execute_placement(sim, selected, destination)
    settle()

    status, pos = verify_placement(sim, selected, destination)
    if status == "VERIFIED":
        print(panel("Placement verified", [f"{selected} is at {_fmt_pos(pos)} inside {destination}."], kind="ok"))
    elif status == "NOT_IN_ZONE":
        print(panel("Placement NOT achieved", [f"{selected} ended at {_fmt_pos(pos)}, outside {destination}."], kind="error"))
    else:
        print(panel("Placement unverified", ["Could not read the object pose; result unknown."], kind="warn"))
    return status, pos


def handle_command(sim, prompt, catalog, ask=input):
    """
    Full pipeline for one text command (terminal mode). Returns a result dict:
      status      - parser/selection status or 'DONE' / 'CANCELLED' / 'PICK_FAILED'
      selected    - chosen object name or None
      destination - zone name / 'throw' / 'safe' / None
      placement   - 'VERIFIED' / 'NOT_IN_ZONE' / 'UNVERIFIED' / None
    """
    result = {'status': None, 'selected': None, 'destination': None,
              'placement': None, 'picked': False, 'camera_image': None}

    analysis = analyze_command(sim, prompt, catalog)
    result['camera_image'] = analysis['camera_image']
    if analysis['status'] not in (OK, AMBIGUOUS):
        result['status'] = analysis['status']
        return result
    parsed, selection = analysis['parsed'], analysis['selection']

    # 4) Ambiguity: never choose silently, ask the user
    if selection.status == AMBIGUOUS:
        lines = [selection.message]
        for i, (name, score) in enumerate(selection.candidate_ranking, 1):
            lines.append(f"  {S(str(i), 'bold', 'cyan')}. {paint_object(name)}  {S(f'{score:.3f}', 'gray')}")
        print(panel("Ambiguous command", lines, kind="warn"))
        pick = ask(f"Choose 1-{len(selection.candidate_ranking)} (Enter = cancel): ").strip()
        if not pick.isdigit() or not 1 <= int(pick) <= len(selection.candidate_ranking):
            print("🚫 Operation cancelled.")
            result['status'] = 'CANCELLED'
            return result
        selection.selected = selection.candidate_ranking[int(pick) - 1][0]

    selected = selection.selected
    result['selected'] = selected
    sel_score = dict(selection.ranking).get(selected, 0.0)
    print(f"\n🎯 {S('Selected', 'bold')}: {paint_object(selected)}  {S(f'(CLIP score {sel_score:.3f})', 'gray')}")

    # Ask user for confirmation
    confirm = ask(f"Pick up {selected}? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes', '']:
        print("🚫 Operation cancelled.")
        result['status'] = 'CANCELLED'
        return result

    # 5) Robot: existing controller + grasp check
    called, grasped = pick_verified(sim, selected)
    result['picked'] = called
    if not grasped:
        result['status'] = 'PICK_FAILED'
        return result

    # 6) Placement zone (from the command, or ask like the original demo)
    destination = parsed.zone or ask_destination(sim, ask)
    result['destination'] = destination

    # 7) Place + verify
    result['placement'], _ = place_held_object(sim, selected, destination)
    result['status'] = 'DONE'
    return result


# ---------------------------------------------------------------------------
# Visual control panel (matplotlib widgets - no extra dependencies)
# ---------------------------------------------------------------------------

PANEL_THEME = {
    'bg': '#0f172a', 'card': '#1e293b', 'edge': '#334155', 'text': '#e2e8f0',
    'muted': '#94a3b8', 'accent': '#38bdf8', 'gold': '#fbbf24',
    'ok': '#22c55e', 'warn': '#f59e0b', 'error': '#ef4444',
}
OBJECT_HEX = {'red': '#ef4444', 'blue': '#3b82f6', 'green': '#22c55e', 'yellow': '#eab308'}
STATUS_KIND = {
    OK: 'ok', AMBIGUOUS: 'warn', 'VERIFIED': 'ok', 'NOT_IN_ZONE': 'error',
    'UNVERIFIED': 'warn', 'PICK_FAILED': 'error', 'BUSY': 'accent', 'IDLE': 'muted',
}


def _object_hex(name):
    color, _ = split_object_name(name or "")
    return OBJECT_HEX.get(color, PANEL_THEME['muted'])


class VisualControlPanel:
    """
    One-window GUI for the interactive pipeline:

        command box -> Analyze -> camera + boxes, sorted CLIP bars, status
                    -> choose zone (or from command) -> Execute -> verified result

    Uses the same functions as the terminal mode (analyze_command,
    pick_verified, place_held_object), so behaviour is identical.
    """

    ZONE_BUTTONS = (('red_zone', 'Red'), ('blue_zone', 'Blue'), ('green_zone', 'Green'),
                    ('yellow_zone', 'Yellow'), ('throw', 'Throw'), ('safe', 'Safe'))
    MAX_CANDIDATES = 3

    def __init__(self, sim, pyplot=None):
        from matplotlib.widgets import TextBox, Button

        self.plt = pyplot or plt
        self.sim = sim
        self.catalog = list(sim.objects.keys())
        self.analysis = None
        self.selected = None
        self.destination = None
        self.busy = False
        T = PANEL_THEME

        fig = self.plt.figure(figsize=(15, 8.6), facecolor=T['bg'])
        try:
            fig.canvas.manager.set_window_title("VLM Robot Color Sorting - Control Panel")
        except Exception:
            pass
        self.fig = fig
        fig.text(0.02, 0.955, "VLM Robot Color Sorting", color=T['text'], fontsize=20, weight='bold')
        fig.text(0.02, 0.925, "CLIP openai/clip-vit-base-patch32  ·  segmentation crops  ·  "
                 "Franka Panda (PyBullet)  ·  pretrained VLM + programmed control, not a trained VLA",
                 color=T['muted'], fontsize=9.5)

        # Camera view
        self.ax_cam = fig.add_axes([0.02, 0.30, 0.56, 0.60])
        # CLIP scores
        self.ax_scores = fig.add_axes([0.71, 0.57, 0.27, 0.33], facecolor=T['card'])
        # Status card
        self.ax_status = fig.add_axes([0.62, 0.30, 0.36, 0.19])
        self.ax_status.set_axis_off()

        # Command row
        self.ax_cmd = fig.add_axes([0.10, 0.19, 0.38, 0.055])
        fig.text(0.02, 0.21, "Command", color=T['text'], fontsize=11, weight='bold')
        self.text_box = TextBox(self.ax_cmd, "", initial="Pick up the red cube",
                                color=T['card'], hovercolor='#273449')
        self.text_box.text_disp.set_color(T['text'])
        self.text_box.text_disp.set_fontsize(12)
        self.text_box.on_submit(lambda _t: self.on_analyze())
        self.btn_analyze = self._button([0.49, 0.19, 0.09, 0.055], "Analyze", T['accent'], self.on_analyze)

        # Ambiguity candidates (hidden until needed)
        fig.text(0.62, 0.225, "", color=T['muted'], fontsize=9)
        self.cand_buttons = []
        for i in range(self.MAX_CANDIDATES):
            ax = fig.add_axes([0.62 + i * 0.122, 0.19, 0.115, 0.055])
            btn = Button(ax, "", color=T['card'], hovercolor='#273449')
            btn.label.set_color(T['text'])
            btn.on_clicked(lambda _e, i=i: self.on_candidate(i))
            ax.set_visible(False)
            self.cand_buttons.append((ax, btn))

        # Zone row
        fig.text(0.02, 0.115, "Place in", color=T['text'], fontsize=11, weight='bold')
        self.zone_buttons = {}
        for i, (dest, label) in enumerate(self.ZONE_BUTTONS):
            if dest not in SPECIAL_DESTINATIONS and dest not in sim.sorting_zones:
                continue
            ax = fig.add_axes([0.10 + i * 0.081, 0.09, 0.075, 0.06])
            btn = Button(ax, label, color=T['card'], hovercolor='#273449')
            btn.label.set_color(OBJECT_HEX.get(label.lower(), T['text']))
            btn.label.set_weight('bold')
            btn.on_clicked(lambda _e, d=dest: self.on_zone(d))
            self.zone_buttons[dest] = (ax, btn)

        self.btn_execute = self._button([0.62, 0.085, 0.36, 0.07], "▶  Pick & Place", T['ok'], self.on_execute)
        fig.text(0.02, 0.03, "Help: name a color (red, blue, green, yellow) and/or shape (cube, sphere, "
                 "cylinder). Optional: '... in the blue zone' / 'throw it away'.  'clip: <text>' = free-form CLIP.",
                 color=T['muted'], fontsize=9)

        self.show_camera(self._safe_capture(), {}, None)
        self.show_scores([], None, [])
        self.set_status("Ready", "Type a command and press Enter or Analyze.", 'IDLE')

    # -- widgets helpers --------------------------------------------------
    def _button(self, rect, label, color, callback):
        from matplotlib.widgets import Button
        ax = self.fig.add_axes(rect)
        btn = Button(ax, label, color=color, hovercolor=PANEL_THEME['text'])
        btn.label.set_color(PANEL_THEME['bg'])
        btn.label.set_weight('bold')
        btn.label.set_fontsize(12)
        btn.on_clicked(lambda _e: callback())
        return btn

    def _redraw(self):
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _safe_capture(self):
        try:
            return self.sim.capture_camera_image()
        except Exception:
            return None

    # -- drawing ----------------------------------------------------------
    def show_camera(self, image, boxes, selected, candidates=()):
        T, ax = PANEL_THEME, self.ax_cam
        ax.clear()
        ax.set_facecolor(T['card'])
        if image is not None and not isinstance(image, str):
            ax.imshow(image)
        else:
            ax.text(0.5, 0.5, "camera image unavailable", color=T['muted'],
                    ha='center', va='center', transform=ax.transAxes)
        import matplotlib.patches as patches
        for name, (x1, y1, x2, y2) in (boxes or {}).items():
            is_sel = name == selected
            edge = T['gold'] if is_sel else _object_hex(name)
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=edge, linewidth=3.5 if is_sel else 1.5,
                                           linestyle='-' if (is_sel or name in candidates) else '--'))
            ax.text(x1, y1 - 6, ("★ " if is_sel else "") + name, fontsize=9, color=T['bg'],
                    weight='bold', bbox=dict(boxstyle="round,pad=0.25", facecolor=edge, edgecolor='none'))
        ax.set_title("Robot camera  ·  segmentation boxes", color=T['text'], fontsize=11, loc='left')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(T['edge'])

    def show_scores(self, ranking, selected, candidates):
        T, ax = PANEL_THEME, self.ax_scores
        ax.clear()
        ax.set_facecolor(T['card'])
        ax.set_title("CLIP similarity (sorted)", color=T['text'], fontsize=11, loc='left')
        if not ranking:
            ax.text(0.5, 0.5, "no scores yet", color=T['muted'], ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            names = [n for n, _ in ranking][::-1]
            vals = [s for _, s in ranking][::-1]
            colors = [_object_hex(n) for n in names]
            alphas = [1.0 if (n == selected or n in candidates) else 0.35 for n in names]
            bars = ax.barh(range(len(names)), vals, color=colors)
            for bar, a, n in zip(bars, alphas, names):
                bar.set_alpha(a)
                if n == selected:
                    bar.set_edgecolor(T['gold']); bar.set_linewidth(2.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([("★ " if n == selected else "") + n for n in names],
                               color=T['text'], fontsize=10)
            for i, v in enumerate(vals):
                ax.text(v + 0.01, i, f"{v:.3f}", va='center', color=T['text'], fontsize=9)
            ax.set_xlim(0, max(1.0, max(vals) * 1.15))
            ax.tick_params(axis='x', colors=T['muted'], labelsize=8)
            ax.set_xlabel("relative softmax over visible crops (not calibrated confidence)",
                          color=T['muted'], fontsize=8)
        for sp in ax.spines.values():
            sp.set_color(T['edge'])

    def set_status(self, title, body, kind_key):
        T, ax = PANEL_THEME, self.ax_status
        ax.clear(); ax.set_axis_off()
        color = T.get(STATUS_KIND.get(kind_key, 'muted'), T['muted'])
        import matplotlib.patches as patches
        ax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.04",
                                            transform=ax.transAxes, facecolor=T['card'],
                                            edgecolor=color, linewidth=2))
        ax.add_patch(patches.Rectangle((0, 0), 0.012, 1, transform=ax.transAxes, color=color))
        ax.text(0.04, 0.80, title, color=color, fontsize=14, weight='bold', transform=ax.transAxes)
        dest = self.destination or "—"
        sel = self.selected or "—"
        ax.text(0.04, 0.50, body, color=T['text'], fontsize=10, transform=ax.transAxes, wrap=True, va='center')
        ax.text(0.04, 0.12, f"Object: {sel}     Destination: {dest}", color=T['muted'],
                fontsize=10, transform=ax.transAxes)

    def _refresh_zone_buttons(self):
        for dest, (ax, btn) in self.zone_buttons.items():
            active = dest == self.destination
            ax.set_facecolor(PANEL_THEME['edge'] if active else PANEL_THEME['card'])
            btn.color = PANEL_THEME['edge'] if active else PANEL_THEME['card']
            for sp in ax.spines.values():
                sp.set_color(PANEL_THEME['gold'] if active else PANEL_THEME['edge'])
                sp.set_linewidth(2.5 if active else 1)

    def _set_candidates(self, names):
        for i, (ax, btn) in enumerate(self.cand_buttons):
            if i < len(names):
                btn.label.set_text(names[i])
                btn.label.set_color(_object_hex(names[i]))
                ax.set_visible(True)
            else:
                ax.set_visible(False)

    # -- callbacks --------------------------------------------------------
    def on_analyze(self):
        if self.busy:
            return
        prompt = self.text_box.text
        self.selected, self.destination = None, None
        self._set_candidates([])
        a = analyze_command(self.sim, prompt, self.catalog)
        self.analysis = a
        parsed, sel = a['parsed'], a['selection']
        if parsed is not None and parsed.zone:
            self.destination = parsed.zone
        if a['camera_image'] is not None:
            self.show_camera(a['camera_image'], a['bounding_boxes'], None, parsed.candidates if parsed else ())

        if sel is None:
            self.show_scores([], None, [])
            msg = parsed.message if parsed and parsed.message else "No usable object crops."
            self.set_status(f"Rejected · {a['status']}", msg + "\nNothing was moved.", 'NOT_IN_ZONE')
        elif a['status'] == AMBIGUOUS:
            names = [n for n, _ in sel.candidate_ranking][:self.MAX_CANDIDATES]
            self.show_scores(sel.ranking, None, parsed.candidates)
            self._set_candidates(names)
            self.set_status("Ambiguous · choose one", sel.message[:160], AMBIGUOUS)
        elif a['status'] == OK:
            self.selected = sel.selected
            self.show_scores(sel.ranking, sel.selected, parsed.candidates)
            self.show_camera(a['camera_image'], a['bounding_boxes'], sel.selected, parsed.candidates)
            note = "" if sel.clip_agrees else f"\nNote: {sel.message}"
            nxt = "Press Pick & Place." if self.destination else "Choose a zone, then Pick & Place."
            self.set_status(f"Selected · {sel.selected}", nxt + note, OK)
        else:
            self.show_scores(sel.ranking, None, parsed.candidates)
            self.set_status(f"Rejected · {a['status']}", sel.message + "\nNothing was moved.", 'NOT_IN_ZONE')
        self._refresh_zone_buttons()
        self._redraw()

    def on_candidate(self, i):
        if self.busy or not self.analysis or not self.analysis['selection']:
            return
        sel = self.analysis['selection']
        if i >= len(sel.candidate_ranking):
            return
        self.selected = sel.candidate_ranking[i][0]
        self._set_candidates([])
        self.show_scores(sel.ranking, self.selected, self.analysis['parsed'].candidates)
        self.show_camera(self.analysis['camera_image'], self.analysis['bounding_boxes'], self.selected)
        self.set_status(f"Selected · {self.selected}", "Chosen by you. Choose a zone, then Pick & Place.", OK)
        self._refresh_zone_buttons()
        self._redraw()

    def on_zone(self, dest):
        if self.busy:
            return
        self.destination = dest
        self._refresh_zone_buttons()
        if self.selected:
            self.set_status(f"Ready · {self.selected} → {dest}", "Press Pick & Place to execute.", OK)
        else:
            self.set_status("Zone chosen", f"Destination {dest}. Analyze a command first.", 'IDLE')
        self._redraw()

    def on_execute(self):
        if self.busy:
            return
        if not self.selected:
            self.set_status("Nothing selected", "Analyze a valid command first.", 'UNVERIFIED')
            self._redraw()
            return
        if not self.destination:
            self.set_status("No zone", "Choose a placement zone first.", 'UNVERIFIED')
            self._redraw()
            return
        self.busy = True
        self.set_status("Robot executing…", f"{self.selected} → {self.destination}", 'BUSY')
        self._redraw()
        try:
            _, grasped = pick_verified(self.sim, self.selected)
            if not grasped:
                self.set_status("Pick failed", "Grasp not verified; nothing was placed.", 'PICK_FAILED')
            else:
                status, pos = place_held_object(self.sim, self.selected, self.destination)
                where = _fmt_pos(pos)
                text = {"VERIFIED": f"Object is inside {self.destination} at {where}.",
                        "NOT_IN_ZONE": f"Object ended at {where}, outside {self.destination}.",
                        }.get(status, "Object pose could not be read; result unknown.")
                self.set_status(f"Placement {status.replace('_', ' ')}", text, status)
            self.show_camera(self._safe_capture(), {}, None)
        finally:
            self.busy = False
            self.selected = None
            self._redraw()

    def show(self):
        self.plt.show()


def run_visual_panel_demo(sim):
    """Open the visual control panel (blocks until the window is closed)."""
    print("\n🖥️  Opening visual control panel... (close the window to finish)")
    VisualControlPanel(sim).show()


def analyze_full_scene_with_prompt(sim, prompt):
    """
    Analyze the full scene image with the prompt instead of individual crops.
    This gives CLIP the complete context to understand spatial relationships
    and potentially count objects.
    
    Args:
        sim: VisionLanguagePandaSimulation instance
        prompt: Text prompt to analyze against full scene
        
    Returns:
        dict: Analysis results including scene description and object recommendations
    """
    print(f"\n🌟 FULL SCENE ANALYSIS MODE")
    print("=" * 40)
    print("Analyzing the complete scene with AI vision...")
    
    # Capture the full scene
    camera_image = sim.capture_camera_image()
    
    # Display the input image (commented out to avoid showing intermediate photos)
    # print("📸 Displaying input scene for analysis...")
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # ax.imshow(camera_image)
    # ax.set_title(f"Full Scene Input for Analysis\nPrompt: '{prompt}'")
    # ax.axis('off')
    
    # Add bounding boxes to show detected objects (commented out)
    # bounding_boxes = sim.get_object_bounding_boxes(camera_image)
    # colors = plt.cm.Set3(range(len(bounding_boxes)))
    
    # for i, (obj_name, bbox) in enumerate(bounding_boxes.items()):
    #     x1, y1, x2, y2 = bbox
    #     color = colors[i]
    #     
    #     # Draw bounding box
    #     import matplotlib.patches as patches
    #     rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
    #                            linewidth=2, edgecolor=color, 
    #                            facecolor='none', alpha=0.8)
    #     ax.add_patch(rect)
    #     
    #     # Add label
    #     ax.text(x1, y1-5, obj_name, fontsize=10, 
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    # plt.tight_layout()
    # plt.show()
    
    # Use CLIP for full scene analysis
    return _analyze_full_scene_clip(sim, camera_image, prompt)


def _analyze_full_scene_clip(sim, camera_image, prompt):
    """Analyze full scene using CLIP model with improved approach."""
    
    print("🔍 Step 1: Analyzing scene context with CLIP...")
    
    # First, let's understand what the prompt is asking for
    # by analyzing it against scene-level descriptions
    scene_understanding_prompts = [
        "a scene with multiple red objects",
        "a scene with multiple colored objects", 
        "objects grouped by color",
        "red objects are the most numerous",
        "green objects are the most numerous",
        "yellow objects are the most numerous",
        prompt  # The original prompt
    ]
    
    # Analyze scene understanding
    inputs = sim.processor(
        text=scene_understanding_prompts,
        images=[camera_image] * len(scene_understanding_prompts),
        return_tensors="pt",
        padding=True
    ).to(sim.device)
    
    with torch.no_grad():
        outputs = sim.model(**inputs)
        scene_logits = outputs.logits_per_image[0]
        scene_probs = torch.softmax(scene_logits, dim=0)
    
    print("📊 Scene Understanding Scores:")
    for i, scene_prompt in enumerate(scene_understanding_prompts):
        score = scene_probs[i].item()
        print(f"   '{scene_prompt}': {score:.3f}")
    
    # Step 2: Now analyze each object individually against the ORIGINAL prompt
    print(f"\n🔍 Step 2: Analyzing each object type against the prompt...")
    
    # Get available objects using new segmentation approach
    print(f"\n🔍 Step 2: Using segmentation-based object detection...")
    _, bounding_boxes, crops = sim.capture_and_process_scene()
    available_objects = list(bounding_boxes.keys())
    
    print(f"   Available objects: {available_objects}")
    
    if not available_objects:
        print("   ❌ No objects detected!")
        return {
            'method': 'full_scene_clip_improved',
            'scene_scores': {},
            'recommended_object': None,
            'confidence': 0.0,
            'analysis': "No objects detected in scene"
        }
    
    print(f"   Processing {len(available_objects)} objects with CLIP...")
    
    # Create specific prompts for each object to better match against the user prompt
    object_specific_prompts = []
    for obj_name in available_objects:
        # Create a contextual prompt that includes the object name and user's intent
        if "most" in prompt.lower() and "object" in prompt.lower():
            # For counting queries, use a comparative prompt
            object_specific_prompts.append(f"the {obj_name.replace('_', ' ')} when asked to {prompt}")
        else:
            # For direct selection, use descriptive prompt
            object_specific_prompts.append(f"a {obj_name.replace('_', ' ')} that matches: {prompt}")
    
    print(f"   Object-specific prompts: {object_specific_prompts}")
    
    # Process with CLIP: we want text-image similarity for each object prompt against the scene
    inputs = sim.processor(
        text=[prompt],  # Single user prompt
        images=[camera_image],  # Single scene image
        return_tensors="pt",
        padding=True
    ).to(sim.device)
    
    print(f"   Input tensor shapes: text=1, images=1")
    
    # Use improved crops from segmentation
    scene_scores = {}
    
    # Analyze each object crop against the user prompt using segmentation-based crops
    for obj_name in available_objects:
        if obj_name in crops:
            # Process this specific segmentation-based crop with the user prompt
            crop_inputs = sim.processor(
                text=[prompt],
                images=[crops[obj_name]],
                return_tensors="pt",
                padding=True
            ).to(sim.device)
            
            with torch.no_grad():
                crop_outputs = sim.model(**crop_inputs)
                # Get similarity score between the prompt and this object crop
                similarity_score = crop_outputs.logits_per_image[0, 0].item()
                # Convert logit to probability-like score
                scene_scores[obj_name] = torch.sigmoid(torch.tensor(similarity_score)).item()
        else:
            scene_scores[obj_name] = 0.0
            print(f"   Warning: No segmentation crop available for {obj_name}")
    
    print(f"   Computed {len(scene_scores)} individual object scores")
    
    print("📊 Object-Level Full Scene Scores:")
    for obj_name, score in scene_scores.items():
        print(f"   {obj_name}: {score:.3f}")
    
    # Step 3: Apply domain knowledge for counting-based prompts
    if "most" in prompt.lower() and ("amount" in prompt.lower() or "objects" in prompt.lower()):
        print("\n🧮 Step 3: Applying counting logic...")
        
        # Count objects by color
        color_counts = {}
        for obj_name in available_objects:
            if 'red' in obj_name.lower():
                color_counts['red'] = color_counts.get('red', 0) + 1
            elif 'blue' in obj_name.lower():
                color_counts['blue'] = color_counts.get('blue', 0) + 1
            elif 'green' in obj_name.lower():
                color_counts['green'] = color_counts.get('green', 0) + 1
            elif 'yellow' in obj_name.lower():
                color_counts['yellow'] = color_counts.get('yellow', 0) + 1
        
        print(f"   Color counts: {color_counts}")
        
        # Find most common color
        if color_counts:
            most_common_color = max(color_counts.items(), key=lambda x: x[1])
            print(f"   Most common color: {most_common_color[0]} ({most_common_color[1]} objects)")
            
            # Boost scores for objects of the most common color
            boosted_scores = scene_scores.copy()
            boost_applied = False
            for obj_name in scene_scores:
                if most_common_color[0] in obj_name.lower():
                    # Significantly boost confidence for correct color
                    original_score = scene_scores[obj_name]
                    boosted_scores[obj_name] = min(0.95, original_score + 0.4)
                    print(f"   Boosted {obj_name}: {original_score:.3f} → {boosted_scores[obj_name]:.3f}")
                    boost_applied = True
            
            if boost_applied:
                print(f"   Applied boost to {most_common_color[0]} objects")
                scene_scores = boosted_scores
            else:
                print(f"   Warning: No objects found matching most common color '{most_common_color[0]}'")
            
            # Debug: show final scores after boosting
            print("📊 Final scores after boosting:")
            for obj_name, score in scene_scores.items():
                print(f"   {obj_name}: {score:.3f}")
    
    # Find best match
    if scene_scores:
        # Sort by score to find the best match
        sorted_scores = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
        best_object = sorted_scores[0]
        
        print(f"\n🏆 Final ranking:")
        for i, (obj_name, score) in enumerate(sorted_scores):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
            print(f"   {rank_emoji} {obj_name}: {score:.3f}")
        
        return {
            'method': 'full_scene_clip_improved',
            'scene_scores': scene_scores,
            'recommended_object': best_object[0],
            'confidence': best_object[1],
            'analysis': f"Improved full scene analysis: {best_object[0]} (confidence: {best_object[1]:.3f})"
        }
    else:
        return {
            'method': 'full_scene_clip_improved',
            'scene_scores': {},
            'recommended_object': None,
            'confidence': 0.0,
            'analysis': "Could not analyze full scene"
        }


def place_object_in_specific_zone(sim, object_name, zone_name):
    """
    Place the currently held object in a specific color sorting zone.
    
    Args:
        sim: VisionLanguagePandaSimulation instance
        object_name (str): Name of the object being placed
        zone_name (str): Name of the target zone ('red_zone', 'blue_zone', etc.)
    """
    if zone_name not in sim.sorting_zones:
        print(f"❌ Unknown zone: {zone_name}")
        safe_place_object(sim, object_name)  # Fallback to safe position
        return
    
    # Get target zone position
    zone_config = sim.sorting_zones[zone_name]
    zone_position = zone_config['position']
    zone_color = zone_name.replace('_zone', '').upper()
    
    print(f"   Placing {object_name} in {zone_color} zone at {zone_position}")
    
    # Move safely to zone approach position (higher up to avoid table collision)
    zone_approach = [zone_position[0], zone_position[1], zone_position[2] + 0.25] 
    print(f"   Moving safely to approach position...")
    sim.safe_move_to_position(zone_approach)
    
    # Lower to zone surface (just above the colored zone)
    zone_target = [zone_position[0], zone_position[1], zone_position[2] + 0.05]  # Higher above zone
    sim.move_to_position(zone_target)
    
    # Release object
    print(f"   Releasing {object_name} in {zone_color} zone...")
    sim.control_gripper(sim.gripper_open_position)
    
    # Retreat with extra height using safe movement
    retreat_pos = [zone_position[0], zone_position[1], zone_position[2] + 0.25]  
    print(f"   Retreating safely...")
    sim.move_to_position(retreat_pos)
    
    # Return to home
    print("   Returning to home position...")
    sim.move_to_position(sim.home_position)


def safe_place_object(sim, object_name):
    """
    Safely place object with higher trajectory to avoid table collisions.
    
    Args:
        sim: VisionLanguagePandaSimulation instance
        object_name (str): Name of the object being placed
    """
    print(f"   Safely placing {object_name}...")
    
    # Move to safe approach position (much higher) using safe movement
    safe_approach = [sim.place_position[0], sim.place_position[1], sim.place_position[2] + 0.30] 
    print(f"   Moving safely to approach position...")
    sim.safe_move_to_position(safe_approach)
    
    # Lower to safe place position (higher than original)
    safe_target = [sim.place_position[0], sim.place_position[1], sim.place_position[2] + 0.05]
    sim.move_to_position(safe_target)
    
    # Release object
    print(f"   Releasing {object_name} safely...")
    sim.control_gripper(sim.gripper_open_position)
    
    # Retreat with extra height
    safe_retreat = [sim.place_position[0], sim.place_position[1], sim.place_position[2] + 0.30] 
    print(f"   Retreating safely...")
    sim.move_to_position(safe_retreat)
    
    # Return to home
    print("   Returning to home position...")
    sim.move_to_position(sim.home_position)


def throw_object_away(sim, object_name):
    """
    Throw the object away with a fast motion and dramatic release!
    
    Args:
        sim: VisionLanguagePandaSimulation instance
        object_name (str): Name of the object being thrown
    """
    print(f"🗑️ Throwing away {object_name} with style!")
    
    # Define throw position (to the side, away from workspace)
    throw_position = [-0.3, 0.5, 0.8]  # To the left and forward, high up
    
    # Fast approach to throw position using safe movement
    print("   🚀 Moving safely to throw position...")
    approach_pos = [throw_position[0], throw_position[1], throw_position[2] + 0.1]
    
    # Use safe movement to get to throw position without hitting table
    sim.safe_move_to_position(approach_pos)
    
    # DRAMATIC RELEASE!
    print("   💨 RELEASING WITH STYLE!")
    sim.control_gripper(sim.gripper_open_position)
    
    # Quick retreat
    print("   ↩️  Quick retreat...")
    retreat_pos = [0.2, 0.0, 0.9]  # Quick retreat position
    sim.move_to_position(retreat_pos)
    
    # Return to home with satisfaction
    print("   🏠 Mission accomplished, returning home...")
    sim.move_to_position(sim.home_position)


if __name__ == "__main__":
    main()
