"""
Generates presentation.pptx for the IRPRJ project.
Run: python make_ppt.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Colour palette ──────────────────────────────────────────
DARK_BLUE   = RGBColor(0x1B, 0x2A, 0x4A)
MED_BLUE    = RGBColor(0x2C, 0x5F, 0x8A)
LIGHT_BLUE  = RGBColor(0x3A, 0x86, 0xC8)
ACCENT_TEAL = RGBColor(0x17, 0xA2, 0xB8)
ACCENT_GREEN = RGBColor(0x28, 0xA7, 0x45)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xF0, 0xF0, 0xF0)
DARK_GRAY   = RGBColor(0x33, 0x33, 0x33)
RED_ACCENT  = RGBColor(0xDC, 0x35, 0x45)


# ── Helpers ─────────────────────────────────────────────────

def _set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_title_bar(slide, text, top=Inches(0), height=Inches(1.05)):
    """Dark blue bar across the top with white title text."""
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), top, Inches(13.33), height
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    tf = bar.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Inches(0.6)
    tf.margin_top = Inches(0.15)
    return bar


def _add_textbox(slide, left, top, width, height, bullets,
                 font_size=18, bold_first=False, color=DARK_GRAY,
                 line_spacing=1.4):
    """Add a text box with bullet points."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, txt in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = txt
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(6)
        p.line_spacing = Pt(font_size * line_spacing)
        if bold_first and i == 0:
            p.font.bold = True
    return txBox


def _add_box_shape(slide, left, top, width, height, text,
                   fill_color=MED_BLUE, font_color=WHITE, font_size=13,
                   bold=True):
    """Rounded rectangle with centered text — for diagrams."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = bold
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    shape.text_frame.paragraphs[0].space_before = Pt(0)
    shape.text_frame.paragraphs[0].space_after = Pt(0)
    return shape


def _add_arrow(slide, start_left, start_top, end_left, end_top):
    """Simple connector arrow."""
    connector = slide.shapes.add_connector(
        1, start_left, start_top, end_left, end_top  # type 1 = straight
    )
    connector.line.color.rgb = DARK_BLUE
    connector.line.width = Pt(2)
    return connector


def _add_thin_line(slide, left, top, width):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, left, top, width, Pt(2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_TEAL
    shape.line.fill.background()
    return shape


# ── Slide builders ──────────────────────────────────────────

def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_slide_bg(slide, DARK_BLUE)

    # Big title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Evaluation Frameworks for Latent\nMulti-Agent Collaboration"
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    # Subtitle
    p2 = tf.add_paragraph()
    p2.text = "Alignment Quality Index, Latent Steering, and Stackelberg Regret"
    p2.font.size = Pt(22)
    p2.font.color.rgb = ACCENT_TEAL
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(16)

    # Separator line
    _add_thin_line(slide, Inches(3.5), Inches(4.0), Inches(6.3))

    # Author / date
    txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(4.3), Inches(11.7), Inches(1))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p3 = tf2.paragraphs[0]
    p3.text = "Based on: \"Latent Collaboration in Multi-Agent Systems\" (Zou et al., 2025)"
    p3.font.size = Pt(16)
    p3.font.color.rgb = RGBColor(0xAA, 0xBB, 0xCC)
    p3.alignment = PP_ALIGN.CENTER

    p4 = tf2.add_paragraph()
    p4.text = "April 2026"
    p4.font.size = Pt(14)
    p4.font.color.rgb = RGBColor(0x88, 0x99, 0xAA)
    p4.alignment = PP_ALIGN.CENTER
    p4.space_before = Pt(8)


def slide_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "Problem Statement")

    bullets = [
        "In heterogeneous multi-agent systems, agents may come from different model families with different alignment priors",
        "Output-level agreement is not enough: agents can sound consistent while occupying different internal alignment regions",
        "During collaboration, this latent mismatch can surface as alignment drift, contradiction,\n"
        "or unsafe coordination under pressure",
        "LatentMAS removes the token bottleneck, but shared latent communication still requires a shared alignment zone",
        "This motivates three linked questions before and during collaboration:",
    ]
    _add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.9), Inches(3.2), bullets,
                 font_size=17)

    q_bullets = [
        "1. Which model should serve as the alignment reference (baseline)?",
        "2. How do we steer other agents into that baseline alignment region?",
        "3. How do we verify that collaboration stays aligned after steering?",
    ]
    _add_textbox(slide, Inches(1.2), Inches(4.6), Inches(10.9), Inches(1.5), q_bullets,
                 font_size=19, bold_first=False, color=MED_BLUE)


def slide_background(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "Background: LatentMAS (Zou et al., 2025)")

    left_bullets = [
        "Latent Thoughts Generation",
        "   Each agent reasons via auto-regressive last-layer\n"
        "   hidden embeddings — no token decoding",
        "",
        "KV-Cache Working Memory Transfer",
        "   Layer-wise KV caches carry both input context\n"
        "   and generated latent thoughts between agents",
        "",
        "Input-Output Alignment",
        "   A linear projection Wa maps hidden states back\n"
        "   to valid input embeddings (training-free)",
    ]
    _add_textbox(slide, Inches(0.7), Inches(1.3), Inches(6.5), Inches(4.5), left_bullets,
                 font_size=16)

    # Key results box
    box = _add_box_shape(slide, Inches(7.8), Inches(1.5), Inches(4.8), Inches(3.8),
                         "", fill_color=RGBColor(0xEE, 0xF4, 0xFA),
                         font_color=DARK_BLUE, font_size=14)

    txBox = slide.shapes.add_textbox(Inches(8.0), Inches(1.6), Inches(4.4), Inches(3.5))
    tf = txBox.text_frame
    tf.word_wrap = True

    lines = [
        ("Key Results (9 benchmarks)", True, Pt(17), DARK_BLUE),
        ("", False, Pt(8), DARK_GRAY),
        ("Up to +14.6% accuracy", False, Pt(16), ACCENT_GREEN),
        ("over single-model baselines", False, Pt(13), DARK_GRAY),
        ("", False, Pt(8), DARK_GRAY),
        ("4x - 4.3x faster inference", False, Pt(16), ACCENT_GREEN),
        ("vs text-based MAS", False, Pt(13), DARK_GRAY),
        ("", False, Pt(8), DARK_GRAY),
        ("70-84% fewer tokens", False, Pt(16), ACCENT_GREEN),
        ("in system-wide output", False, Pt(13), DARK_GRAY),
    ]
    for i, (text, bold, size, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = size
        p.font.bold = bold
        p.font.color.rgb = color
        p.alignment = PP_ALIGN.CENTER


def slide_gap(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "Research Gap Identified")

    # Gap 1
    _add_box_shape(slide, Inches(0.7), Inches(1.5), Inches(5.8), Inches(2.2),
                   "", fill_color=RGBColor(0xFD, 0xF2, 0xF2),
                   font_color=DARK_GRAY, font_size=14)
    txBox = slide.shapes.add_textbox(Inches(0.9), Inches(1.6), Inches(5.4), Inches(2.0))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Gap 1: Baseline Selection"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = RED_ACCENT
    p2 = tf.add_paragraph()
    p2.text = ("The paper states that the best-aligned model's latent space "
               "becomes the steering target, but provides no systematic method "
               "to evaluate and select this baseline model.")
    p2.font.size = Pt(15)
    p2.font.color.rgb = DARK_GRAY
    p2.space_before = Pt(8)

    # Gap 2
    _add_box_shape(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(2.2),
                   "", fill_color=RGBColor(0xFD, 0xF2, 0xF2),
                   font_color=DARK_GRAY, font_size=14)
    txBox2 = slide.shapes.add_textbox(Inches(7.2), Inches(1.6), Inches(5.4), Inches(2.0))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p3 = tf2.paragraphs[0]
    p3.text = "Gap 2: Runtime Alignment Monitoring"
    p3.font.size = Pt(20)
    p3.font.bold = True
    p3.font.color.rgb = RED_ACCENT
    p4 = tf2.add_paragraph()
    p4.text = ("Theorem 3.3 proves lossless information transfer under ideal conditions, "
               "but in practice agents may drift, hit suboptimal equilibria, or develop "
               "adversarial representations. No runtime metric exists to detect this.")
    p4.font.size = Pt(15)
    p4.font.color.rgb = DARK_GRAY
    p4.space_before = Pt(8)

    # Our contributions box
    _add_box_shape(slide, Inches(2.5), Inches(4.3), Inches(8.3), Inches(1.8),
                   "", fill_color=RGBColor(0xE8, 0xF5, 0xE9),
                   font_color=DARK_GRAY, font_size=14)
    txBox3 = slide.shapes.add_textbox(Inches(2.7), Inches(4.4), Inches(7.9), Inches(1.6))
    tf3 = txBox3.text_frame
    tf3.word_wrap = True
    p5 = tf3.paragraphs[0]
    p5.text = "Our Contributions"
    p5.font.size = Pt(20)
    p5.font.bold = True
    p5.font.color.rgb = ACCENT_GREEN
    p5.alignment = PP_ALIGN.CENTER
    contribs = [
        "1. Alignment Quality Index (AQI) — principled baseline model selection across 5 dimensions",
        "2. Latent Steering Adapter — maps non-baseline agent latents toward the baseline alignment region",
        "3. Stackelberg Regret Evaluator — runtime monitoring of collaboration quality and alignment drift",
        "4. PettingZoo Integration — end-to-end validation of baseline selection, steering, and post-steering evaluation",
    ]
    for c in contribs:
        p = tf3.add_paragraph()
        p.text = c
        p.font.size = Pt(14)
        p.font.color.rgb = DARK_GRAY
        p.space_before = Pt(4)
        p.alignment = PP_ALIGN.CENTER


def slide_aqi(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "Contribution 1: Alignment Quality Index (AQI)")

    dimensions = [
        ("Harmlessness  (30%)", "Refuses dangerous/unethical requests.\n15 probes: hacking, weapons, phishing, stalking...\nKeyword-based refusal detection"),
        ("Helpfulness  (25%)", "Gives substantive, accurate answers.\n15 probes: science, CS, economics topics.\nEmbedding similarity + response length"),
        ("Honesty  (20%)", "Admits uncertainty for unknowable questions;\nanswers factual questions correctly.\n15 probes: future prediction, private info, facts"),
        ("Consistency  (15%)", "Paraphrased questions yield same answer.\n15 groups x 3 paraphrases each.\nPairwise cosine similarity of embeddings"),
        ("Instruction Fidelity  (10%)", "Follows formatting constraints.\n15 probes: bullet lists, JSON, haiku, tables...\n15 different format-type parsers"),
    ]

    colors = [RED_ACCENT, MED_BLUE, RGBColor(0xE6, 0x7E, 0x22),
              ACCENT_TEAL, RGBColor(0x6F, 0x42, 0xC1)]

    y_start = Inches(1.3)
    box_h = Inches(0.92)
    gap = Inches(0.08)

    for i, ((title, desc), color) in enumerate(zip(dimensions, colors)):
        y = y_start + i * (box_h + gap)

        # Color accent bar
        accent = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5), y, Inches(0.12), box_h
        )
        accent.fill.solid()
        accent.fill.fore_color.rgb = color
        accent.line.fill.background()

        # Title
        txT = slide.shapes.add_textbox(Inches(0.8), y, Inches(3.0), box_h)
        tf = txT.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = color

        # Description
        txD = slide.shapes.add_textbox(Inches(3.9), y, Inches(8.8), box_h)
        tf2 = txD.text_frame
        tf2.word_wrap = True
        for j, line in enumerate(desc.split("\n")):
            pp = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
            pp.text = line
            pp.font.size = Pt(13)
            pp.font.color.rgb = DARK_GRAY

    # Bottom note
    _add_textbox(slide, Inches(0.7), Inches(6.4), Inches(12), Inches(0.5),
                 ["Total: 75 hand-crafted probes across 5 dimensions  |  Weighted composite score selects the baseline model"],
                 font_size=14, color=MED_BLUE)


def slide_aqi_arch(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "AQI as the Baseline Selection Objective")

    # Flow diagram boxes
    bw = Inches(2.1)
    bh = Inches(0.7)
    y = Inches(2.2)
    gap = Inches(0.55)

    boxes_top = [
        ("Probe family\n(75 probes, 5 dims)", Inches(0.5)),
        ("Dimension scores\nH, Help, Hon, Cons, IF", Inches(0.5) + bw + gap),
        ("Weighted alignment\nfunctional AQI(m)", Inches(0.5) + 2*(bw + gap)),
        ("Ranking over candidate\nmodels", Inches(0.5) + 3*(bw + gap)),
        ("Baseline reference\nb = argmax AQI(m)", Inches(0.5) + 4*(bw + gap)),
    ]
    colors_top = [LIGHT_BLUE, MED_BLUE, MED_BLUE, DARK_BLUE, ACCENT_GREEN]

    for (text, x), col in zip(boxes_top, colors_top):
        _add_box_shape(slide, x, y, bw, bh, text, fill_color=col, font_size=12)

    # Arrows between boxes
    for i in range(4):
        x_start = boxes_top[i][1] + bw
        x_end = boxes_top[i+1][1]
        mid_y = y + bh / 2
        line = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW, x_start, mid_y - Pt(8), x_end - x_start, Pt(16)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = DARK_BLUE
        line.line.fill.background()

    # Scoring details below
    score_y = Inches(3.6)
    score_boxes = [
        ("Harmlessness\nsafety refusal\nsignal", RED_ACCENT),
        ("Helpfulness\nsemantic utility\nsignal", MED_BLUE),
        ("Honesty\nuncertainty or\nfactual accuracy", RGBColor(0xE6, 0x7E, 0x22)),
        ("Consistency\nparaphrase\ninvariance", ACCENT_TEAL),
        ("Instruction Fidelity\nconstraint\nsatisfaction", RGBColor(0x6F, 0x42, 0xC1)),
    ]
    sbw = Inches(2.1)
    sbh = Inches(1.0)
    for i, (text, col) in enumerate(score_boxes):
        x = Inches(0.5) + i * (sbw + Inches(0.35))
        _add_box_shape(slide, x, score_y, sbw, sbh, text, fill_color=col, font_size=11)

    # Label
    _add_textbox(slide, Inches(0.5), Inches(4.9), Inches(12), Inches(0.6),
                 ["AQI(m) = 0.30 H(m) + 0.25 Help(m) + 0.20 Hon(m) + 0.15 Cons(m) + 0.10 IF(m)"],
                 font_size=15, color=DARK_BLUE)

    # Demo note
    _add_textbox(slide, Inches(0.5), Inches(5.6), Inches(12), Inches(1.0),
                 ["Interpretation: AQI induces an ordering over candidate collaborators before interaction begins",
                  "The highest-AQI model becomes the alignment reference that defines the steering target region"],
                 font_size=14, color=DARK_GRAY)


def slide_sr(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "Contribution 2: Stackelberg Regret Evaluator")

    # Formula
    _add_textbox(slide, Inches(0.7), Inches(1.2), Inches(12), Inches(0.6),
                 ["SR = V_leader(pi_leader, BR_intended) - V_leader(pi_leader, pi_observed)"],
                 font_size=20, color=DARK_BLUE, bold_first=True)
    _add_textbox(slide, Inches(0.7), Inches(1.7), Inches(12), Inches(0.4),
                 ["SR = 0: aligned  |  SR > 0: followers misaligned  |  SR < 0: over-performing (rare)"],
                 font_size=14, color=DARK_GRAY)

    # Four sub-metrics in boxes
    metrics = [
        ("Behavioral SR\n(weight: 0.35)",
         "Action-space gap between\nleader's expected value under\nintended BR vs observed\nfollower policy",
         RED_ACCENT),
        ("Latent SR\n(weight: 0.30)",
         "Representation-space gap:\nlearned projection maps\nz_leader to expected z_follower;\nL2 distance = misalignment",
         MED_BLUE),
        ("Nash Deviation Index\n(weight: 0.20)",
         "Equilibrium stability check:\ncan any follower improve by\ndeviating? NDI > 0 means\nunstable / unconverged",
         ACCENT_TEAL),
        ("Latent Cone Membership\n(weight: 0.15)",
         "Geometric check: Mahalanobis\nellipsoid fitted on aligned\nepisodes; fraction of follower\nlatents inside = cone rate",
         RGBColor(0x6F, 0x42, 0xC1)),
    ]

    bw = Inches(2.85)
    bh_title = Inches(0.65)
    bh_desc = Inches(1.3)
    y_title = Inches(2.4)
    y_desc = y_title + bh_title + Inches(0.05)

    for i, (title, desc, color) in enumerate(metrics):
        x = Inches(0.5) + i * (bw + Inches(0.2))
        _add_box_shape(slide, x, y_title, bw, bh_title, title,
                       fill_color=color, font_size=13)
        _add_box_shape(slide, x, y_desc, bw, bh_desc, desc,
                       fill_color=RGBColor(0xFA, 0xFA, 0xFA), font_color=DARK_GRAY,
                       font_size=12, bold=False)

    # Additional feature
    _add_textbox(slide, Inches(0.7), Inches(5.1), Inches(12), Inches(1.4),
                 ["Composite SR = 0.35 * max(0, Behavioral SR) + 0.30 * Latent SR + 0.20 * tanh(NDI) + 0.15 * (1 - cone rate)",
                  "Per-Follower BRG localizes which collaborator has drifted away from equilibrium",
                  "Rolling-window drift tracking turns alignment from a static score into a temporal process"],
                 font_size=15, color=DARK_GRAY)


def slide_steering_math(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "Latent Steering: Theoretical & Mathematical Basis")

    _add_textbox(slide, Inches(0.7), Inches(1.15), Inches(12), Inches(0.55),
                 ["Goal: given a baseline model b, move each collaborator i into the same alignment region before cooperation begins"],
                 font_size=18, color=DARK_BLUE, bold_first=True)

    _add_box_shape(slide, Inches(0.6), Inches(1.9), Inches(3.9), Inches(0.8),
                   "Step 1: Baseline selection\nb = argmax_m AQI(m)",
                   fill_color=ACCENT_TEAL, font_size=16)

    _add_box_shape(slide, Inches(4.75), Inches(1.9), Inches(3.9), Inches(0.8),
                   "Step 2: Learn adapter T_i\nfrom aligned latent pairs",
                   fill_color=MED_BLUE, font_size=16)

    _add_box_shape(slide, Inches(8.9), Inches(1.9), Inches(3.8), Inches(0.8),
                   "Step 3: Apply steering\nbefore action decoding",
                   fill_color=ACCENT_GREEN, font_size=16)

    _add_textbox(slide, Inches(0.8), Inches(3.0), Inches(12), Inches(0.7),
                 ["Training objective:  L_i = E_t || T_i(z_i^t) - z_b^t ||_2^2 + lambda ||W_i - I||_F^2"],
                 font_size=18, color=DARK_BLUE, bold_first=True)

    _add_textbox(slide, Inches(0.8), Inches(3.7), Inches(12), Inches(0.7),
                 ["Steering rule:  z_i^steered = (1 - alpha) z_i + alpha T_i(z_i)"],
                 font_size=18, color=DARK_BLUE, bold_first=True)

    theory_points = [
        "The regularizer keeps the adapter close to identity, preserving the source policy manifold and reducing breakage",
        "Interpolation parameter alpha controls the strength of movement toward the baseline alignment zone",
        "Aligned calibration trajectories provide paired samples (z_i^t, z_b^t) for each non-baseline agent",
        "Success is measured not by latent overlap alone, but by post-steering improvements in SR and cone membership",
    ]
    _add_textbox(slide, Inches(0.9), Inches(4.5), Inches(11.9), Inches(1.6),
                 theory_points, font_size=15, color=DARK_GRAY)

    _add_box_shape(slide, Inches(2.6), Inches(6.1), Inches(8.2), Inches(0.6),
                   "Desired effect: SR_drifted > SR_steered and cone_rate_steered > cone_rate_drifted",
                   fill_color=RGBColor(0xE8, 0xF5, 0xE9), font_color=DARK_BLUE,
                   font_size=16, bold=False)


def slide_pettingzoo(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "Contribution 3: End-to-End Validation in PettingZoo")

    _add_textbox(slide, Inches(0.7), Inches(1.2), Inches(12), Inches(0.5),
                 ["Environment: MPE simple_spread_v3 — three cooperative agents with heterogeneous alignment profiles"],
                 font_size=17, color=DARK_BLUE, bold_first=True)

    # Pipeline steps
    steps = [
        ("1. Train", "Train cooperative agents\nand attach candidate\nalignment profiles"),
        ("2. Select Baseline", "Run AQI over agent\nprofiles and choose the\nhighest-AQI baseline"),
        ("3. Fit Steering", "Use aligned trajectories\nto learn latent adapters\nfor non-baseline agents"),
        ("4. Stress Test", "Inject latent drift,\nthen compare unsteered\nand steered behavior"),
    ]

    step_colors = [LIGHT_BLUE, ACCENT_TEAL, MED_BLUE, DARK_BLUE]
    bw = Inches(2.6)
    bh = Inches(1.5)
    y = Inches(2.0)

    for i, ((title, desc), col) in enumerate(zip(steps, step_colors)):
        x = Inches(0.5) + i * (bw + Inches(0.3))
        _add_box_shape(slide, x, y, bw, Inches(0.5), title, fill_color=col, font_size=14)
        _add_box_shape(slide, x, y + Inches(0.55), bw, Inches(1.0), desc,
                       fill_color=LIGHT_GRAY, font_color=DARK_GRAY, font_size=12, bold=False)

    # Three scenarios
    _add_textbox(slide, Inches(0.7), Inches(4.1), Inches(12), Inches(0.5),
                 ["Four evaluation regimes:"], font_size=17, color=DARK_BLUE, bold_first=True)

    scenarios = [
        ("A: Aligned", "Nominal collaboration.\nReference case with low SR\nand high cone rate.", ACCENT_GREEN),
        ("B: Drifted", "One follower is pushed out\nof the baseline region.\nSR should increase.", RGBColor(0xE6, 0x7E, 0x22)),
        ("C: Drifted + Steered", "Same perturbation, but the\nadapter pulls latents back.\nSR should decrease.", MED_BLUE),
        ("D: Fully Random", "Untrained policies give a\nlower-bound baseline for\ncollaboration quality.", RED_ACCENT),
    ]
    sw = Inches(3.0)
    for i, (title, desc, col) in enumerate(scenarios):
        x = Inches(0.35) + i * (sw + Inches(0.2))
        _add_box_shape(slide, x, Inches(4.7), sw, Inches(0.45), title, fill_color=col, font_size=13)
        _add_box_shape(slide, x, Inches(5.2), sw, Inches(0.9), desc,
                       fill_color=LIGHT_GRAY, font_color=DARK_GRAY, font_size=12, bold=False)


def slide_summary(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, DARK_BLUE)

    # Title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Summary & Takeaways"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    _add_thin_line(slide, Inches(3.5), Inches(1.35), Inches(6.3))

    # Three-stage pipeline
    stages = [
        ("Pre-Collaboration\nAQI Module",
         "Evaluate candidate agents\nacross 5 alignment dimensions.\nSelect the highest-AQI model\nas the reference alignment\nzone for collaboration.",
         ACCENT_TEAL),
        ("Alignment Intervention\nLatent Steering",
         "Learn residual adapters that\nmove non-baseline latents\ntoward the baseline region\nwhile preserving useful\nsource behavior.",
         ACCENT_GREEN),
        ("During Collaboration\nSR Evaluator",
         "Measure whether steering\nactually improves cooperation\nvia behavioral, latent, game-\ntheoretic, and geometric\nsignals.",
         MED_BLUE),
    ]

    bw = Inches(3.5)
    bh_top = Inches(0.8)
    bh_bot = Inches(2.0)
    y_top = Inches(1.8)
    y_bot = y_top + bh_top + Inches(0.1)

    for i, (title, desc, col) in enumerate(stages):
        x = Inches(0.75) + i * (bw + Inches(0.35))
        _add_box_shape(slide, x, y_top, bw, bh_top, title,
                       fill_color=col, font_size=15)
        _add_box_shape(slide, x, y_bot, bw, bh_bot, desc,
                       fill_color=RGBColor(0x24, 0x3B, 0x5E), font_color=WHITE,
                       font_size=13, bold=False)

    # Bottom key points
    txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(5.2), Inches(11.7), Inches(1.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    points = [
        "Core claim: alignment must be established in latent space before collaboration, not inferred only from outputs",
        "AQI supplies the reference model, latent steering supplies the intervention, and SR supplies the verification signal",
        "The resulting pipeline is theoretical, measurable, and directly extensible to richer multi-agent benchmarks",
    ]
    for i, pt in enumerate(points):
        pp = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        pp.text = pt
        pp.font.size = Pt(15)
        pp.font.color.rgb = ACCENT_TEAL
        pp.alignment = PP_ALIGN.CENTER
        pp.space_before = Pt(6)


# ── Main ────────────────────────────────────────────────────

def main():
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_problem(prs)
    slide_background(prs)
    slide_gap(prs)
    slide_aqi(prs)
    slide_aqi_arch(prs)
    slide_sr(prs)
    slide_steering_math(prs)
    slide_pettingzoo(prs)
    slide_summary(prs)

    out = "presentation.pptx"
    prs.save(out)
    print(f"Saved {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
