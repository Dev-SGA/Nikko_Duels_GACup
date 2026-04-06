import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Duel Map Analysis")

st.title("Offensive Duel Map Analysis - Multiple Matches")
st.caption("Click on the icons on the pitch to inspect the corresponding event.")

# ==========================
# Duel Data
# ==========================
duel_matches_data = {
    "Vs San Jose": [
        ("OFFENSIVE DUEL WON", 58.01, 22.88, "videos/D2 - SJ.mp4"),
        ("OFFENSIVE DUEL LOST", 83.61, 38.17, "videos/D1 - SJ.mp4"),
        ("OFFENSIVE DUEL LOST", 91.25, 77.07, "videos/D3 - SJ.mp4"),
    ],
    "Vs Copehagen": [
        ("OFFENSIVE DUEL LOST", 60.50, 13.90, "videos/D2 - CP.mp4"),
        ("OFFENSIVE DUEL LOST", 97.07, 26.20, "videos/D1 - CP.mp4"),
    ],
    "Vs Sporting": [
        ("OFFENSIVE DUEL WON", 75.79, 29.69, "videos/D3 - SP.mp4"),
        ("OFFENSIVE DUEL WON", 75.46, 24.21, "videos/D2 - SP.mp4"),
        ("OFFENSIVE DUEL LOST", 41.38, 9.75, "videos/D1 - SP.mp4"),
        ("OFFENSIVE DUEL LOST", 89.26, 12.74, "videos/D4 - SP.mp4"),
    ],
}

# ==========================
# Touch Data
# ==========================
touches_matches_data = {
    "Vs San Jose": [
        (66.98, 20.72),
        (85.10, 38.84),
        (93.08, 31.69),
        (105.05, 31.69),
        (108.71, 36.18),
        (84.10, 74.08),
    ],
    "Vs Copehagen": [
        (57.67, 13.74),
        (97.90, 3.10),
        (88.59, 19.22),
        (92.58, 28.36),
        (94.41, 25.37),
        (86.60, 49.31),
        (84.60, 57.79),
        (68.48, 66.10),
        (93.25, 68.43),
        (98.57, 74.74),
    ],
    "Vs Sporting": [
        (49.86, 7.75),
        (58.17, 15.56),
        (54.18, 25.70),
        (70.47, 34.68),
        (80.28, 17.23),
        (92.75, 11.41),
    ],
}

# ==========================
# Create DataFrames
# ==========================
duel_dfs_by_match = {}
for match_name, events in duel_matches_data.items():
    df_match = pd.DataFrame(events, columns=["type", "x", "y", "video"])
    df_match["number"] = np.arange(1, len(df_match) + 1)
    duel_dfs_by_match[match_name] = df_match

touch_dfs_by_match = {}
for match_name, events in touches_matches_data.items():
    df_touch = pd.DataFrame(events, columns=["x", "y"])
    df_touch["number"] = np.arange(1, len(df_touch) + 1)
    touch_dfs_by_match[match_name] = df_touch

df_duels_all = pd.concat(duel_dfs_by_match.values(), ignore_index=True)
duel_full_data = {"All Games": df_duels_all}
duel_full_data.update(duel_dfs_by_match)

df_touches_all = pd.concat(touch_dfs_by_match.values(), ignore_index=True)
touch_full_data = {"All Games": df_touches_all}
touch_full_data.update(touch_dfs_by_match)

# ==========================
# Helpers
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""

def get_style(event_type, has_video):
    event_type = event_type.upper()

    if "OFFENSIVE" in event_type:
        if "WON" in event_type:
            return "o", (0.10, 0.85, 0.10, 0.95), 110, 0.8
        if "LOST" in event_type:
            alpha = 0.95 if has_video else 0.75
            return "x", (0.95, 0.15, 0.15, alpha), 120, 3.0

    if "DEFENSIVE" in event_type:
        if "WON" in event_type:
            return "s", (0.00, 0.60, 0.00, 0.90), 110, 0.8
        if "LOST" in event_type:
            alpha = 0.90 if has_video else 0.65
            return "D", (0.70, 0.00, 0.00, alpha), 110, 2.5

    if "AERIAL" in event_type:
        if "WON" in event_type:
            return "^", (0.20, 0.50, 0.95, 0.90), 120, 0.8
        if "LOST" in event_type:
            return "v", (0.55, 0.20, 0.85, 0.85), 120, 0.8

    if "FOULED" in event_type:
        return "P", (1.00, 0.80, 0.00, 1.00), 130, 0.8

    return "o", (0.5, 0.5, 0.5, 0.8), 90, 0.5

def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)

    is_duel = df["type"].str.contains("DUEL|AERIAL", case=False, na=False)
    is_won = df["type"].str.contains("WON", case=False, na=False)
    is_offensive = df["type"].str.contains("OFFENSIVE", case=False, na=False)

    all_duels = df[is_duel]
    total_duels = len(all_duels)
    won_duels = all_duels[is_won].shape[0]
    lost_duels = total_duels - won_duels
    duel_rate = (won_duels / total_duels * 100) if total_duels > 0 else 0

    off_duels = df[is_offensive & is_duel]
    off_total = len(off_duels)
    off_wins = off_duels[is_won].shape[0]
    off_rate = (off_wins / off_total * 100) if off_total > 0 else 0

    final_third_mask = df["x"] > 80
    final_third_duels = df[final_third_mask & is_duel]
    final_third_total = len(final_third_duels)
    final_third_wins = final_third_duels[is_won].shape[0]
    final_third_lost = final_third_total - final_third_wins
    final_third_rate = (final_third_wins / final_third_total * 100) if final_third_total > 0 else 0

    fouls = len(df[df["type"].str.contains("FOULED", case=False, na=False)])

    return {
        "total": total,
        "duel_total": total_duels,
        "duel_wins": won_duels,
        "duel_lost": lost_duels,
        "duel_rate": duel_rate,
        "off_total": off_total,
        "off_wins": off_wins,
        "off_rate": off_rate,
        "final_third_total": final_third_total,
        "final_third_wins": final_third_wins,
        "final_third_lost": final_third_lost,
        "final_third_rate": final_third_rate,
        "fouls": fouls,
    }

# ==========================
# Sidebar
# ==========================
st.sidebar.header("Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(duel_full_data.keys()), index=0)

st.sidebar.divider()

filter_duel_type = st.sidebar.multiselect(
    "Duel Type",
    ["Offensive", "Defensive", "Aerial", "Other"],
    default=["Offensive"]
)

st.sidebar.divider()
st.sidebar.caption("The duel map is filtered by the selected options above.")

# ==========================
# Filter Data
# ==========================
df_duels = duel_full_data[selected_match].copy()

if not all(x in filter_duel_type for x in ["Offensive", "Defensive", "Aerial", "Other"]):
    mask = pd.Series([False] * len(df_duels), index=df_duels.index)

    if "Offensive" in filter_duel_type:
        mask |= df_duels["type"].str.contains("OFFENSIVE", case=False, na=False)
    if "Defensive" in filter_duel_type:
        mask |= df_duels["type"].str.contains("DEFENSIVE", case=False, na=False)
    if "Aerial" in filter_duel_type:
        mask |= df_duels["type"].str.contains("AERIAL", case=False, na=False)
    if "Other" in filter_duel_type:
        mask |= ~df_duels["type"].str.contains("OFFENSIVE|DEFENSIVE|AERIAL", case=False, na=False)

    df_duels = df_duels[mask]

df_touches = touch_full_data[selected_match].copy()
stats = compute_stats(duel_full_data[selected_match])

# ==========================
# Top Row: Duel Map + Touch Heatmap
# ==========================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Interactive Duel Map")

    pitch_duel = Pitch(
        pitch_type="statsbomb",
        pitch_color="#f8f8f8",
        line_color="#4a4a4a"
    )
    fig_duel, ax_duel = pitch_duel.draw(figsize=(8, 5.6))

    for _, row in df_duels.iterrows():
        has_vid = has_video_value(row["video"])
        marker, color, size, lw = get_style(row["type"], has_vid)
        edge_color = "black" if has_vid else "none"

        pitch_duel.scatter(
            row["x"], row["y"],
            marker=marker,
            s=size,
            color=color,
            edgecolors=edge_color,
            linewidths=lw,
            ax=ax_duel,
            zorder=3
        )

    ax_duel.annotate(
        "",
        xy=(70, 83),
        xytext=(50, 83),
        arrowprops=dict(arrowstyle="->", color="#4a4a4a", lw=1.5)
    )
    ax_duel.text(
        60, 86,
        "Attack Direction",
        ha="center",
        va="center",
        fontsize=9,
        color="#4a4a4a",
        fontweight="bold"
    )

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Offensive Duel Won",
            markerfacecolor=(0.10, 0.85, 0.10, 0.95),
            markersize=10,
            linestyle="None"
        ),
        Line2D(
            [0], [0],
            marker="x",
            color=(0.95, 0.15, 0.15, 0.95),
            label="Offensive Duel Lost",
            markersize=10,
            markeredgewidth=2.5,
            linestyle="None"
        ),
    ]

    legend = ax_duel.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        fontsize="small",
        title="Match Events",
        title_fontsize="medium",
        labelspacing=1.2,
        borderpad=1.0,
        framealpha=0.95
    )
    legend.get_title().set_fontweight("bold")

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_obj = Image.open(buf)

    click = streamlit_image_coordinates(img_obj, width=700)
    plt.close(fig_duel)

with col_right:
    st.subheader("Touch Heatmap")

    pitch_hm = Pitch(
        pitch_type="statsbomb",
        pitch_color="#6BB36B",
        line_color="white"
    )
    fig_hm, ax_hm = pitch_hm.draw(figsize=(8, 5.6))

    if not df_touches.empty:
        pitch_hm.kdeplot(
            df_touches["x"],
            df_touches["y"],
            ax=ax_hm,
            cmap="Reds",
            shade=True,
            levels=100,
            alpha=0.7
        )

        pitch_hm.scatter(
            df_touches["x"],
            df_touches["y"],
            ax=ax_hm,
            color="black",
            s=18,
            alpha=0.65,
            zorder=3
        )

    ax_hm.set_title(f"Touch Heatmap - {selected_match}", fontsize=12)

    arrow = FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05),
        transform=fig_hm.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2,
        color="#333333"
    )
    fig_hm.patches.append(arrow)

    fig_hm.text(
        0.5, 0.03,
        "Attack Direction",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333"
    )

    st.pyplot(fig_hm, use_container_width=False)
    plt.close(fig_hm)

# ==========================
# Click Interaction
# ==========================
selected_event = None

if click is not None:
    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)

    mpl_pixel_y = real_h - pixel_y
    coords = ax_duel.transData.inverted().transform((pixel_x, mpl_pixel_y))
    field_x, field_y = coords[0], coords[1]

    df_sel = df_duels.copy()
    df_sel["dist"] = np.sqrt((df_sel["x"] - field_x) ** 2 + (df_sel["y"] - field_y) ** 2)

    RADIUS = 5
    candidates = df_sel[df_sel["dist"] < RADIUS]

    if not candidates.empty:
        selected_event = candidates.loc[candidates["dist"].idxmin()]

# ==========================
# Bottom Row: Event Details + Statistics
# ==========================
st.divider()

col_bottom_left, col_bottom_right = st.columns(2)

with col_bottom_left:
    st.subheader("Event Details")

    if selected_event is not None:
        st.success(f"Selected Event: {selected_event['type']}")
        st.info(f"Position: X: {selected_event['x']:.2f}, Y: {selected_event['y']:.2f}")

        if has_video_value(selected_event["video"]):
            try:
                st.video(selected_event["video"])
            except Exception:
                st.error(f"Video file not found: {selected_event['video']}")
        else:
            st.warning("No video footage is available for this event.")
    else:
        st.info("Select a marker on the duel map to view event details.")

with col_bottom_right:
    st.subheader("Performance Statistics")

    # ── Overall Duels ──
    st.markdown("**Overall Duels**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", stats["duel_total"])
    c2.metric("Won", stats["duel_wins"])
    c3.metric("Lost", stats["duel_lost"])
    c4.metric("Success %", f"{stats['duel_rate']:.1f}%")

    st.divider()

    # ── Final Third Duels ──
    st.markdown("**Final Third Duels**")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total", stats["final_third_total"])
    f2.metric("Won", stats["final_third_wins"])
    f3.metric("Lost", stats["final_third_lost"])
    f4.metric("Success %", f"{stats['final_third_rate']:.1f}%")
