# patterns_to_video.py
# Save Gray code and 4-step sine (phase-shifted) patterns into a single video.

import math
import argparse
import numpy as np
import cv2

# ---------- Gray code ----------
def generate_graycode_patterns(w, h):
    nx = int(math.ceil(math.log2(w)))
    ny = int(math.ceil(math.log2(h)))
    xs = np.arange(w, dtype=np.uint32)
    ys = np.arange(h, dtype=np.uint32)
    gx = xs ^ (xs >> 1)
    gy = ys ^ (ys >> 1)
    pats = []
    # X bits, MSB -> LSB, with complements
    for k in range(nx - 1, -1, -1):
        col = ((gx >> k) & 1).astype(np.uint8) * 255
        img = np.repeat(col[np.newaxis, :], h, axis=0)
        pats += [img, 255 - img]
    # Y bits, MSB -> LSB, with complements
    for k in range(ny - 1, -1, -1):
        row = ((gy >> k) & 1).astype(np.uint8) * 255
        img = np.repeat(row[:, np.newaxis], w, axis=1)
        pats += [img, 255 - img]
    black = np.zeros((h, w), np.uint8)
    white = np.full((h, w), 255, np.uint8)
    return pats, black, white

# ---------- 4-step sine patterns ----------
def gen_sine_patterns(W, H, periods, nphase=4, axis='x', gamma=None):
    if axis == 'x':
        u = np.linspace(0, 2*np.pi*periods, W, endpoint=False)[None, :]
        u = np.repeat(u, H, axis=0)
    else:
        u = np.linspace(0, 2*np.pi*periods, H, endpoint=False)[:, None]
        u = np.repeat(u, W, axis=1)

    pats = []
    for k in range(nphase):
        phi = 2*np.pi * (k / nphase)          # 0, 90°, 180°, 270°
        img = 0.5 + 0.5*np.cos(u + phi)       # [0..1]
        u8  = (img * 255.0 + 0.5).astype(np.uint8)
        if gamma is not None:
            x = np.arange(256, dtype=np.float32)/255.0
            lut = np.clip((x ** (1.0/gamma))*255.0 + 0.5, 0, 255).astype(np.uint8)
            u8 = cv2.LUT(u8, lut)
        pats.append(u8)
    return pats

# ---------- video writer helpers ----------
def to_bgr(img_u8):
    if img_u8.ndim == 2:
        return cv2.merge([img_u8, img_u8, img_u8])
    if img_u8.shape[2] == 3:
        return img_u8
    raise ValueError("Unsupported image shape")

def write_sequence(out, frames, hold, annotate=None, font_scale=0.6, thickness=1):
    for i, f in enumerate(frames):
        frame = to_bgr(f)
        if annotate:
            text = annotate(i, f.shape[1], f.shape[0])
            if text:
                cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                cv2.putText(frame, text, (11, 27), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        for _ in range(hold):
            out.write(frame)

def main():
    ap = argparse.ArgumentParser(description="Save Gray-code and 4-step sine patterns to video.")
    ap.add_argument("--width",  type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fps",    type=float, default=30.0)
    ap.add_argument("--hold",   type=int, default=6, help="frames per pattern")
    ap.add_argument("--outfile", default="patterns.mp4")
    ap.add_argument("--codec",   default="mp4v", help="fourcc, e.g., mp4v, XVID, H264")
    ap.add_argument("--sine_x_periods", type=int, default=64)
    ap.add_argument("--sine_y_periods", type=int, default=48)
    ap.add_argument("--gamma", type=float, default=None, help="inverse gamma for projector linearization, e.g., 2.2")
    ap.add_argument("--no_labels", action="store_true", help="disable on-frame text labels")
    args = ap.parse_args()

    W, H = args.width, args.height
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.outfile, fourcc, args.fps, (W, H), True)
    if not out.isOpened():
        raise RuntimeError("Could not open video writer")

    # Gray code set
    gray, black, white = generate_graycode_patterns(W, H)
    seq_gray = [black] + gray + [black, white]

    def ann_gray(i, w, h):
        if args.no_labels: return None
        if i == 0: return "GRAY: black"
        if i == len(seq_gray)-2: return "GRAY: black (end)"
        if i == len(seq_gray)-1: return "GRAY: white"
        # Within gray sequence: even indices (after initial black) are X/MSB->LSB and complements, then Y/MSB->LSB and complements.
        return "GRAY: pattern %d" % i

    write_sequence(out, seq_gray, args.hold, ann_gray)

    # Sine X (4-step)
    sine_x = gen_sine_patterns(W, H, args.sine_x_periods, nphase=4, axis='x', gamma=args.gamma)
    def ann_sx(i, w, h):
        if args.no_labels: return None
        return f"SINE X: period={args.sine_x_periods}, phase {i}*90deg"
    write_sequence(out, sine_x, args.hold, ann_sx)

    # Sine Y (4-step)
    sine_y = gen_sine_patterns(W, H, args.sine_y_periods, nphase=4, axis='y', gamma=args.gamma)
    def ann_sy(i, w, h):
        if args.no_labels: return None
        return f"SINE Y: period={args.sine_y_periods}, phase {i}*90deg"
    write_sequence(out, sine_y, args.hold, ann_sy)

    out.release()

if __name__ == "__main__":
    main()
