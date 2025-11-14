from camera_internals.fisheye_KDxi import *
from generate_projector_camera_mapping.mapping_pipeline import *
from generate_projector_camera_mapping.refine_mapx_mapy import *
from generate_projector_camera_mapping.perspective_checker_with_exp_maps import *
from stimulus.warp_moving_circle_fish import *


# --- chessboard calibration (K, D, xi) ---
IMG_GLOB    = r"calibration.pictures/1800 U 501m NIR-07XC0/automatic.chessboard/*.png"
SQUARE_SIZE = 1.0
PATTERNS    = [(6, 9), (9, 6)]
TOP_N_BAD   = 10

SUBPIX_CRIT = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-5,
)
CALIB_CRIT = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    300,
    1e-10,
)

# calculate_KDxi(
#     IMG_GLOB=IMG_GLOB,
#     SQUARE_SIZE=SQUARE_SIZE,
#     PATTERNS=PATTERNS,
#     TOP_N_BAD=TOP_N_BAD,
#     SUBPIX_CRIT=SUBPIX_CRIT,
#     CALIB_CRIT=CALIB_CRIT,
# )

# --- shared camera / projector config ---
CAMTYPE = "alvium"   # "rotpy" or "alvium"
PROJ_W  = 1280
PROJ_H  = 800

# --- mapping (structured light) ---
MODE       = "sine_hybrid"  # "gray" or "sine_hybrid"
PERIODS_X  = 128            # ~12.5 px period for 800 px width
PERIODS_Y  = 96             # ~12.5 px period for 600 px height
NPHASE     = 15
AVG_PER    = 5              # increase to 3–5 if needed
GAMMA_INV  = None           # set 2.2 to apply inverse gamma to sines

# execute_mapping(
#     CAMTYPE=CAMTYPE,
#     PROJ_W=PROJ_W,
#     PROJ_H=PROJ_H,
#     MODE=MODE,
#     PERIODS_X=PERIODS_X,
#     PERIODS_Y=PERIODS_Y,
#     NPHASE=NPHASE,
#     AVG_PER=AVG_PER,
#     GAMMA_INV=GAMMA_INV,
# )

# --- sparse refinement of mapx/mapy ---
STEP    = 7       # projector grid spacing in px
DOT_R   = 5        # radius of projected dot (currently unused)
EXP_MS  = 3.0
GAIN_DB = 0.0
SETTLE  = 0.01    # seconds to wait after updating projector
SIZE    = 10

refine_mapx_mapy(
    STEP=STEP,
    DOT_R=DOT_R,
    EXP_MS=EXP_MS,
    GAIN_DB=GAIN_DB,
    SETTLE=SETTLE,
    SIZE=SIZE,
    CAMTYPE=CAMTYPE,
)

# --- perspective checker with experimental maps ---
ts  = time.strftime("%Y%m%d_%H%M%S")
OUT = fr"D:/screen.calibration/debug/perspective.checker.with.exp.maps/{ts}"

run_perspective_checker(
    CAMTYPE=CAMTYPE,
    MODE=MODE,
    PERIODS_X=PERIODS_X,
    PERIODS_Y=PERIODS_Y,
    NPHASE=NPHASE,
    AVG_PER=AVG_PER,
    GAMMA_INV=GAMMA_INV,
    PROJ_W=PROJ_W,
    PROJ_H=PROJ_H,
    OUT=OUT,
)

# --- moving circle, fisheye-rectified recording ---
EXPOSURE_MS   = 7.0
OSC_FREQ_HZ   = 0.25
SAVE_UNDISTORT = True
OMNI_CFG_DIR  = r"D:\screen.calibration/configs/fisheye.config"
OUTDIR        = r"D:\screen.calibration/debug/moving.circle.fish.alvium"
ZOOM_UNDISTORT = 0.1
TIMEOUT_STIM   = 20.0
USE_EXPERIMENTAL_MAP = True

DOT_RADIUS_FRAC = 0.05
DOT_Y_FRAC      = 0.65
OSC_AMPL_FRAC   = 0.8

RAW_PATH       = os.path.join(OUTDIR, "osc_circle_alvium_raw.mp4")
UND_PATH       = os.path.join(OUTDIR, "osc_circle_alvium_undist.mp4")
LAST_PROJ_PNG  = os.path.join(OUTDIR, "osc_circle_last_projector_frame.png")

warp_circle_oscillate_and_record(
    PROJ_W=PROJ_W,
    PROJ_H=PROJ_H,
    EXPOSURE_MS=EXPOSURE_MS,
    GAIN_DB=GAIN_DB,
    OSC_FREQ_HZ=OSC_FREQ_HZ,
    SAVE_UNDISTORT=SAVE_UNDISTORT,
    OMNI_CFG_DIR=OMNI_CFG_DIR,
    OUTDIR=OUTDIR,
    ZOOM_UNDISTORT=ZOOM_UNDISTORT,
    TIMEOUT_STIM=TIMEOUT_STIM,
    USE_EXPERIMENTAL_MAP=USE_EXPERIMENTAL_MAP,
    DOT_RADIUS_FRAC=DOT_RADIUS_FRAC,
    DOT_Y_FRAC=DOT_Y_FRAC,
    OSC_AMPL_FRAC=OSC_AMPL_FRAC,
    RAW_PATH=RAW_PATH,
    UND_PATH=UND_PATH,
    LAST_PROJ_PNG=LAST_PROJ_PNG,
)
