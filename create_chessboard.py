# Generate a 10×7 inner-corner chessboard (11×8 squares) as an A4 PDF,
# keeping square edge length fixed. Auto-chooses portrait/landscape to fit.
#
# Dependencies: matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----------------- CONFIG -----------------
INNER_CORNERS_X = 10          # corners along X (columns)
INNER_CORNERS_Y = 7           # corners along Y (rows)
SQUARE_EDGE_MM  = 25.0        # square edge length in millimetres (kept fixed)
MARGIN_MM       = 5.0         # page margin (white border) in millimetres
OUTPUT_PDF      = f"chessboard_{INNER_CORNERS_X}x{INNER_CORNERS_Y}_A4.pdf"
# ------------------------------------------

SQUARES_X = INNER_CORNERS_X
SQUARES_Y = INNER_CORNERS_Y

# A4 page size in mm
A4_PORTRAIT  = (210.0, 297.0)  # (width, height)
A4_LANDSCAPE = (297.0, 210.0)

board_w_mm = SQUARES_X * SQUARE_EDGE_MM
board_h_mm = SQUARES_Y * SQUARE_EDGE_MM

def fits(page_w, page_h, bw, bh, m):
    return (bw + 2*m <= page_w) and (bh + 2*m <= page_h)

# Choose orientation that fits with fixed square size
if fits(*A4_LANDSCAPE, board_w_mm, board_h_mm, MARGIN_MM):
    page_w_mm, page_h_mm = A4_LANDSCAPE
elif fits(*A4_PORTRAIT, board_w_mm, board_h_mm, MARGIN_MM):
    page_w_mm, page_h_mm = A4_PORTRAIT
else:
    # If neither fits, place centered in landscape (it will clip at page edges)
    page_w_mm, page_h_mm = A4_LANDSCAPE

# Figure size in inches
MM_PER_INCH = 25.4
fig_w_in = page_w_mm / MM_PER_INCH
fig_h_in = page_h_mm / MM_PER_INCH

fig = plt.figure(figsize=(fig_w_in, fig_h_in))
ax = plt.axes([0, 0, 1, 1])
ax.set_xlim(0, page_w_mm)
ax.set_ylim(0, page_h_mm)
ax.set_aspect('equal')
ax.axis('off')

# White background
ax.add_patch(Rectangle((0, 0), page_w_mm, page_h_mm,
                       facecolor='white', edgecolor='none'))

# Center the board with at least MARGIN_MM
origin_x = max((page_w_mm - board_w_mm) / 2.0, MARGIN_MM)
origin_y = max((page_h_mm - board_h_mm) / 2.0, MARGIN_MM)

# Draw checkerboard; black at (row+col) even
for r in range(SQUARES_Y):
    for c in range(SQUARES_X):
        if (r + c) % 2 == 0:
            x = origin_x + c * SQUARE_EDGE_MM
            y = origin_y + r * SQUARE_EDGE_MM
            ax.add_patch(Rectangle((x, y),
                                   SQUARE_EDGE_MM, SQUARE_EDGE_MM,
                                   facecolor='black', edgecolor='black', linewidth=0.2))

# Thin border around board
ax.add_patch(Rectangle((origin_x, origin_y),
                       board_w_mm, board_h_mm,
                       facecolor='none', edgecolor='black', linewidth=0.5))

plt.savefig(OUTPUT_PDF, format="pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
