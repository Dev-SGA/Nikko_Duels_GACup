import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Duel Map Analysis")

st.title("Offensive Duel Map Analysis - Multiple Matches")
st.caption("Click on the icons on the pitch to inspect the corresponding event.")

# ==========================
# Data Setup
# ==========================
matches_data = {
    "Vs San Jose": [
        ("OFFENSIVE DUEL WON", 58.01, 22.88, None),
        ("OFFENSIVE DUEL LOST", 83.61, 38.17, None),
        ("OFFENSIVE DUEL LOST", 91.25, 77.07, None),
    ],
    "Vs Copehagen": [
        ("OFFENSIVE DUEL LOST", 60.50, 13.90, None),
        ("OFFENSIVE DUEL LOST", 97.07, 26.20, None),
    ],
    "Vs Sporting": [
        ("OFFENSIVE DUEL WON", 75.79, 29.69, None),
        ("OFFENSIVE DUEL WON", 75.46, 24.21, None),
        ("OFFENSIVE DUEL LOST", 41.38, 9.75, None),
        ("OFFENSIVE DUEL LOST", 89.26, 12.74, None),
    ],
}

# Create DataFrames for each match and combined
dfs_by_match = {}
for match_name, events in matches_data.items():
    df_match = pd.DataFrame(events, columns=["type", "x", "y", "video"])
    df_match["number"] = np.arange(1, len(df_match) + 1)
    dfs_by_match[match_name] = df_match

# All games combined
df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All Games": df_all}
full_data.update(dfs_by_match)

# ==========================
# Styling
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""

def get_style(event_type, has_video):
    """Returns marker, color (rgba), size, and linewidth based on event type"""
    event_type = event_type.upper()

    # Offensive duels
    if "OFFENSIVE" in event_type:
        if "WON" in event_type:
            return 'o', (0.10, 0.85, 0.10, 0.95), 110, 0.8
        if "LOST" in event_type:
            alpha = 0.95 if has_video else 0.75
            return 'x', (0.95, 0.15, 0.15, alpha), 120, 3.0

    # Defensive duels
    if "DEFENSIVE" in event_type:
        if "WON" in event_type:
            return 's', (0.00, 0.60, 0.00, 0.90), 110, 0.8
        if "LOST" in event_type:
            alpha = 0.90 if has_video else 0.65
            return 'D', (0.70, 0.00, 0.00, alpha), 110, 2.5

    # Aerial duels
    if "AERIAL" in event_type:
        if "WON" in event_type:
            return '^', (0.20, 0.50, 0.95, 0.90), 120, 0.8
        if "LOST" in event_type:
            return 'v', (0.55, 0.20, 0.85, 0.85), 120, 0.8

    # Other events
    if "FOULED" in event_type:
        return 'P', (1.00, 0.80, 0.00, 1.00), 130, 0.8

    return 'o', (0.5, 0.5, 0.5, 0.8), 90, 0.5

# ==========================
# Stats
# ==========================
def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)

    is_duel = df['type'].str.contains('DUEL|AERIAL', case=False, na=False)
    is_won = df['type'].str.contains('WON', case=False, na=False)
    is_offensive = df['type'].str.contains('OFFENSIVE', case=False, na=False)
    is_defensive = df['type'].str.contains('DEFENSIVE', case=False, na=False)

    all_duels = df[is_duel]
    total_duels = len(all_duels)
    won_duels = all_duels[is_won].shape[0]
    duel_rate = (won_duels / total_duels * 100) if total_duels > 0 else 0

    off_duels = df[is_offensive & is_duel]
    off_total = len(off_duels)
    off_wins = off_duels[is_won].shape[0]
    off_rate = (off_wins / off_total * 100) if off_total > 0 else 0

    def_duels = df[is_defensive & is_duel]
    def_total = len(def_duels)
    def_wins = def_duels[is_won].shape[0]
    def_rate = (def_wins / def_total * 100) if def_total > 0 else 0

    aerial_duels = df[df['type'].str.contains('AERIAL', case=False, na=False)]
    aerial_total = len(aerial_duels)
    aerial_wins = aerial_duels[is_won].shape[0]
    aerial_rate = (aerial_wins / aerial_total * 100) if aerial_total > 0 else 0

    left_mask = df['y'] < 26.6
    left_duels = df[left_mask & is_duel]
    left_total = len(left_duels)
    left_wins = left_duels[is_won].shape[0]
    left_rate = (left_wins / left_total * 100) if left_total > 0 else 0

    central_mask = (df['y'] >= 26.6) & (df['y'] <= 53.3)
    central_duels = df[central_mask & is_duel]
    central_total = len(central_duels)
    central_wins = central_duels[is_won].shape[0]
    central_rate = (central_wins / central_total * 100) if central_total > 0 else 0

    right_mask = df['y'] > 53.3
    right_duels = df[right_mask & is_duel]
    right_total = len(right_duels)
    right_wins = right_duels[is_won].shape[0]
    right_rate = (right_wins / right_total * 100) if right_total > 0 else 0

    final_third_mask = df['x'] > 80
    final_third_duels = df[final_third_mask & is_duel]
    final_third_total = len(final_third_duels)
    final_third_wins = final_third_duels[is_won].shape[0]
    final_third_rate = (final_third_wins / final_third_total * 100) if final_third_total > 0 else 0

    fouls = len(df[df['type'].str.contains('FOULED', case=False, na=False)])

    return {
        "total": total,
        "duel_total": total_duels,
        "duel_wins": won_duels,
        "duel_rate": duel_rate,
        "off_total": off_total,
        "off_wins": off_wins,
        "off_rate": off_rate,
        "def_total": def_total,
        "def_wins": def_wins,
        "def_rate": def_rate,
        "aerial_total": aerial_total,
        "aerial_wins": aerial_wins,
        "aerial_rate": aerial_rate,
        "left_total": left_total,
        "left_wins": left_wins,
        "left_rate": left_rate,
        "central_total": central_total,
        "central_wins": central_wins,
        "central_rate": central_rate,
        "right_total": right_total,
        "right_wins": right_wins,
        "right_rate": right_rate,
        "final_third_total": final_third_total,
        "final_third_wins": final_third_wins,
        "final_third_rate": final_third_rate,
        "fouls": fouls,
    }

# ==========================
# Sidebar Configuration
# ==========================
st.sidebar.header("Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

filter_duel_type = st.sidebar.multiselect(
    "Duel Type",
    ["Offensive", "Defensive", "Aerial", "Other"],
    default=["Offensive"]
)

st.sidebar.divider()
st.sidebar.caption("The pitch map is filtered by the selected options above.")

# Get selected data
df = full_data[selected_match].copy()

# Apply duel type filter
if not all(x in filter_duel_type for x in ["Offensive", "Defensive", "Aerial", "Other"]):
    mask = pd.Series([False] * len(df), index=df.index)

    if "Offensive" in filter_duel_type:
        mask |= df['type'].str.contains('OFFENSIVE', case=False, na=False)
    if "Defensive" in filter_duel_type:
        mask |= df['type'].str.contains('DEFENSIVE', case=False, na=False)
    if "Aerial" in filter_duel_type:
        mask |= df['type'].str.contains('AERIAL', case=False, na=False)
    if "Other" in filter_duel_type:
        mask |= ~df['type'].str.contains('OFFENSIVE|DEFENSIVE|AERIAL', case=False, na=False)

    df = df[mask]

# Compute stats from the selected full match data
stats = compute_stats(full_data[selected_match])

# ==========================
# Main Layout
# ==========================
col_map, col_vid = st.columns([1, 1])

with col_map:
    st.subheader("Interactive Pitch Map")

    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='#f8f8f8',
        line_color='#4a4a4a'
    )
    fig, ax = pitch.draw(figsize=(10, 7))

    for _, row in df.iterrows():
        has_vid = has_video_value(row["video"])
        marker, color, size, lw = get_style(row["type"], has_vid)

        edge_color = 'black' if has_vid else 'none'

        pitch.scatter(
            row.x, row.y,
            marker=marker,
            s=size,
            color=color,
            edgecolors=edge_color,
            linewidths=lw,
            ax=ax,
            zorder=3
        )

    # Attack Arrow
    ax.annotate(
        '',
        xy=(70, 83),
        xytext=(50, 83),
        arrowprops=dict(arrowstyle='->', color='#4a4a4a', lw=1.5)
    )
    ax.text(
        60, 86,
        "Attack Direction",
        ha='center',
        va='center',
        fontsize=9,
        color='#4a4a4a',
        fontweight='bold'
    )

    # Legend
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Offensive Duel Won',
            markerfacecolor=(0.10, 0.85, 0.10, 0.95),
            markersize=10,
            linestyle='None'
        ),
        Line2D(
            [0], [0],
            marker='x',
            color=(0.95, 0.15, 0.15, 0.95),
            label='Offensive Duel Lost',
            markersize=10,
            markeredgewidth=2.5,
            linestyle='None'
        ),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor='white',
        edgecolor='#333333',
        fontsize='small',
        title="Match Events",
        title_fontsize='medium',
        labelspacing=1.2,
        borderpad=1.0,
        framealpha=0.95
    )
    legend.get_title().set_fontweight('bold')

    # Convert plot to image for coordinate tracking
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_obj = Image.open(buf)

    click = streamlit_image_coordinates(img_obj, width=700)

# ==========================
# Interaction Logic
# ==========================
selected_event = None

if click is not None:
    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)

    mpl_pixel_y = real_h - pixel_y
    coords = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
    field_x, field_y = coords[0], coords[1]

    df_sel = df.copy()
    df_sel["dist"] = np.sqrt((df_sel["x"] - field_x) ** 2 + (df_sel["y"] - field_y) ** 2)

    RADIUS = 5
    candidates = df_sel[df_sel["dist"] < RADIUS]

    if not candidates.empty:
        selected_event = candidates.loc[candidates["dist"].idxmin()]

# ==========================
# Video Display & Stats
# ==========================
with col_vid:
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
        st.info("Select a marker on the pitch to view event details.")

    st.divider()
    st.subheader("Performance Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Overall Duels",
        f"{stats['duel_wins']}/{stats['duel_total']}",
        f"{stats['duel_rate']:.1f}% Success"
    )
    col2.metric(
        "Duels in Final Third",
        f"{stats['final_third_wins']}/{stats['final_third_total']}",
        f"{stats['final_third_rate']:.1f}% Success"
    )
    col3.metric("Fouls Suffered", stats["fouls"])
