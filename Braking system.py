"""
Fuzzy Logic Automatic Braking System (Mamdani inference)
========================================================

Inputs  : Distance to Obstacle (0-50 m), Relative Speed (0-120 km/h),
          Road Friction coefficient (0-1)
Output  : Brake Force (0-100 %)

Pipeline: fuzzification -> rule evaluation (AND = min) -> implication (clip)
          -> aggregation (max) -> defuzzification (centroid).

Run:
    python fuzzy_braking_system.py
    python fuzzy_braking_system.py 12 85 0.30
    python fuzzy_braking_system.py --plot

Only NumPy is required. Matplotlib is needed only for --plot.
"""

import sys
import numpy as np

# ----------------------------------------------------------------------
# 1. Membership functions
# ----------------------------------------------------------------------

def tri(x, a, b, c):
    """Triangular membership. Handles shoulders where a == b or b == c."""
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12),
                    np.where(x >= b, 1.0, 0.0))
    right = np.where(c > b, (c - x) / (c - b + 1e-12),
                     np.where(x <= b, 1.0, 0.0))
    return np.clip(np.minimum(left, right), 0.0, 1.0)


DISTANCE = {
    "Near": (0, 0, 15),
    "Medium": (10, 25, 40),
    "Far": (30, 50, 50),
}

SPEED = {
    "Slow": (0, 0, 40),
    "Moderate": (20, 60, 100),
    "Fast": (80, 120, 120),
}

FRICTION = {
    "Low": (0, 0, 0.4),
    "Medium": (0.2, 0.5, 0.8),
    "High": (0.6, 1.0, 1.0),
}

BRAKE = {
    "None": (0, 0, 15),
    "Light": (10, 30, 50),
    "Moderate": (35, 55, 75),
    "Hard": (65, 90, 100),
}

# Output universe sampled at 1001 points over 0-100%
Y = np.linspace(0, 100, 1001)

# ----------------------------------------------------------------------
# 2. Rule base
# Format: (distance, speed, friction, brake)
# None means the input is not used by the rule.
# ----------------------------------------------------------------------

RULES = [
    ("Near", "Fast", "Low", "Hard"),        # R1
    ("Near", "Fast", None, "Hard"),         # R2
    ("Near", "Slow", None, "Moderate"),     # R3
    ("Medium", "Fast", None, "Moderate"),   # R4
    ("Medium", None, "Low", "Moderate"),    # R5
    ("Far", None, None, "None"),            # R6
    (None, None, "Low", "Moderate"),        # R7
    ("Near", "Slow", "Low", "Hard"),        # R8
]

# ----------------------------------------------------------------------
# 3. Fuzzy inference
# ----------------------------------------------------------------------

def mu(variable, term, x):
    """Membership degree of crisp value x in one term."""
    return float(tri(x, *variable[term]))


def fuzzify(distance, speed, friction):
    """Convert crisp inputs into membership degrees."""
    return {
        "distance": {
            t: mu(DISTANCE, t, distance) for t in DISTANCE
        },
        "speed": {
            t: mu(SPEED, t, speed) for t in SPEED
        },
        "friction": {
            t: mu(FRICTION, t, friction) for t in FRICTION
        },
    }


def evaluate_rules(memberships):
    """Calculate rule strengths using minimum for AND."""
    fired = []

    for i, (d, s, f, out) in enumerate(RULES, start=1):
        degrees = []

        if d is not None:
            degrees.append(memberships["distance"][d])

        if s is not None:
            degrees.append(memberships["speed"][s])

        if f is not None:
            degrees.append(memberships["friction"][f])

        fired.append((f"R{i}", min(degrees), out))

    return fired


def aggregate(fired):
    """Clip output membership functions and aggregate using maximum."""
    agg = np.zeros_like(Y)

    for _, w, out in fired:
        if w > 0:
            agg = np.maximum(
                agg,
                np.minimum(w, tri(Y, *BRAKE[out]))
            )

    return agg


def centroid(agg):
    """Calculate the centre of gravity of the aggregated fuzzy set."""
    area = agg.sum()

    if area > 0:
        return float((Y * agg).sum() / area)

    return 0.0


def weighted_average(fired):
    """Calculate a classroom-style weighted average."""
    strongest = {}

    for _, w, out in fired:
        strongest[out] = max(strongest.get(out, 0.0), w)

    total = sum(strongest.values())

    if total == 0:
        return 0.0

    return sum(
        w * BRAKE[t][1] for t, w in strongest.items()
    ) / total


def brake_force(distance, speed, friction):
    """Return brake force percentage using centroid defuzzification."""
    fired = evaluate_rules(fuzzify(distance, speed, friction))
    return centroid(aggregate(fired))


# ----------------------------------------------------------------------
# 4. Reporting helpers
# ----------------------------------------------------------------------

def explain(distance, speed, friction):
    """Print each stage of inference for one sensor reading."""
    m = fuzzify(distance, speed, friction)
    fired = evaluate_rules(m)
    agg = aggregate(fired)

    print(
        f"Sensor snapshot: distance = {distance} m, "
        f"speed = {speed} km/h, friction = {friction}"
    )

    print("\nFuzzification (non-zero degrees):")

    for var, terms in m.items():
        shown = ", ".join(
            f"{t} = {v:.3f}"
            for t, v in terms.items()
            if v > 0
        )
        print(f"  {var:<9}: {shown}")

    print("\nRules fired:")

    for name, w, out in fired:
        if w > 0:
            print(f"  {name}: w = {w:.3f} -> {out}")

    if not any(w > 0 for _, w, _ in fired):
        print("  none")

    print(f"\nBrake force, centroid           : {centroid(agg):.1f} %")
    print(
        f"Brake force, weighted average   : "
        f"{weighted_average(fired):.1f} %"
    )


def threshold_rule(distance, limit=10.0):
    """Baseline: brake at 100% below a specified distance."""
    return 100.0 if distance < limit else 0.0


def scenarios():
    cases = [
        ("Near, slow, dry", 5, 15, 0.90),
        ("Near, fast, icy", 5, 110, 0.10),
        ("Worked example", 12, 85, 0.30),
        ("Far, fast, dry", 45, 100, 0.90),
    ]

    print(
        f"\n{'Scenario':<18}{'Dist (m)':>9}"
        f"{'Speed':>8}{'Fric':>7}{'Brake %':>10}"
    )

    for name, d, s, f in cases:
        print(
            f"{name:<18}{d:>9}{s:>8}{f:>7.2f}"
            f"{brake_force(d, s, f):>10.1f}"
        )


def sweep_distance(speed=85, friction=0.30):
    print(
        f"\nDistance sweep at speed {speed} km/h, "
        f"friction {friction}"
    )
    print(f"{'Dist (m)':>9}{'Fuzzy %':>10}{'Threshold %':>13}")

    for d in range(0, 51, 5):
        print(
            f"{d:>9}{brake_force(d, speed, friction):>10.1f}"
            f"{threshold_rule(d):>13.0f}"
        )

    a = brake_force(14.9, speed, friction)
    b = brake_force(15.1, speed, friction)

    print(
        f"\nSensor jitter 14.9 m vs 15.1 m: "
        f"fuzzy {a:.1f} % vs {b:.1f} % "
        f"(change {abs(a - b):.1f} points)"
    )


# ----------------------------------------------------------------------
# 5. Optional plots
# ----------------------------------------------------------------------

def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        (
            "Distance to Obstacle (m)",
            DISTANCE,
            np.linspace(0, 50, 501),
        ),
        (
            "Relative Speed (km/h)",
            SPEED,
            np.linspace(0, 120, 601),
        ),
        (
            "Road Friction coefficient",
            FRICTION,
            np.linspace(0, 1, 501),
        ),
        (
            "Brake Force (%)",
            BRAKE,
            Y,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    for ax, (title, var, xs) in zip(axes.ravel(), specs):
        for term, params in var.items():
            ax.plot(
                xs,
                tri(xs, *params),
                label=term,
                linewidth=2,
            )

        ax.set_title(title)
        ax.set_ylabel("Membership")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig("membership_functions.png", dpi=150)
    plt.close(fig)

    # Aggregated output for the worked example
    fired = evaluate_rules(fuzzify(12, 85, 0.30))
    agg = aggregate(fired)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(Y, agg, alpha=0.3)
    ax.plot(Y, agg, linewidth=2)

    ax.axvline(
        centroid(agg),
        color="black",
        label=f"Centroid {centroid(agg):.1f} %",
    )

    ax.axvline(
        weighted_average(fired),
        color="gray",
        linestyle="--",
        label=f"Weighted average {weighted_average(fired):.1f} %",
    )

    ax.set_xlabel("Brake force (%)")
    ax.set_ylabel("Membership")
    ax.set_title("Aggregated output: 12 m, 85 km/h, friction 0.30")
    ax.legend()
    fig.tight_layout()
    fig.savefig("aggregated_output.png", dpi=150)
    plt.close(fig)

    # Fuzzy vs threshold across distance
    ds = np.linspace(0, 50, 101)
    fz = [brake_force(d, 85, 0.30) for d in ds]
    th = [threshold_rule(d) for d in ds]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ds, fz, linewidth=2.5, label="Fuzzy controller")

    ax.step(
        ds,
        th,
        where="post",
        linewidth=2.5,
        color="gray",
        label="Threshold rule (d < 10 m)",
    )

    ax.set_xlabel("Distance to obstacle (m)")
    ax.set_ylabel("Brake force (%)")
    ax.set_title("Speed 85 km/h, friction 0.30")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fuzzy_vs_threshold.png", dpi=150)
    plt.close(fig)

    print(
        "\nSaved: membership_functions.png, "
        "aggregated_output.png, fuzzy_vs_threshold.png"
    )


# ----------------------------------------------------------------------
# 6. Entry point
# ----------------------------------------------------------------------

def main(argv):
    plot = "--plot" in argv
    nums = [a for a in argv if a != "--plot"]

    if len(nums) == 3:
        explain(float(nums[0]), float(nums[1]), float(nums[2]))
    else:
        explain(12, 85, 0.30)
        scenarios()
        sweep_distance()

    if plot:
        make_plots()


if __name__ == "__main__":
    main(sys.argv[1:])