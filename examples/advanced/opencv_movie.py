"""
An example that uses a function from an external C library (OpenCV in this
case). It uses the C++ implementation with ``cpp_standalone``. Optional
standalone devices can be selected with ``BRIAN2_STANDALONE_DEVICE`` and
``BRIAN2_STANDALONE_MODULE``; the Rust Device uses a periodic Python callback
and the OpenCV Python bindings instead of requiring native OpenCV headers.

Set ``BRIAN2_OPENCV_CAMERA`` to a camera index (usually ``0``) to use a live
camera instead of the example movie. Camera runs are bounded to 120 frames by
default; ``BRIAN2_OPENCV_FRAMES`` changes that limit. The result window accepts
space/click to pause or resume, ``r`` to restart, and ``q``/escape to close.
Set ``BRIAN2_OPENCV_SIZE``, e.g. ``64x36``, to resize frames before simulation,
and ``BRIAN2_OPENCV_GUI=0`` for unattended runs.

The C++ path needs native OpenCV headers and libraries as well as the Python
bindings. The Rust path only needs the Python bindings. The original C++ path
was tested on 64 bit Linux in a conda environment with packages from the
``conda-forge`` channels (opencv 3.4.4, x264 1!152.20180717, ffmpeg 4.1).
"""
import importlib
import json
import os
import urllib.request

import cv2  # Import OpenCV2

from brian2 import *

defaultclock.dt = 1*ms
prefs.logging.std_redirection = False
standalone_device = os.environ.get("BRIAN2_STANDALONE_DEVICE", "cpp_standalone")
standalone_module = os.environ.get("BRIAN2_STANDALONE_MODULE")
standalone_directory = os.environ.get("BRIAN2_STANDALONE_DIRECTORY")
if standalone_module:
    importlib.import_module(standalone_module)
if standalone_device == "cpp_standalone":
    prefs.codegen.target = "cython"
    options = {"clean": True}
else:
    options = {}
if standalone_directory:
    options["directory"] = standalone_directory
set_device(standalone_device, **options)

configured_video = os.environ.get("BRIAN2_OPENCV_VIDEO")
configured_camera = os.environ.get("BRIAN2_OPENCV_CAMERA")
configured_frames = os.environ.get("BRIAN2_OPENCV_FRAMES")
if configured_video and configured_camera is not None:
    raise ValueError(
        "Set only one of BRIAN2_OPENCV_VIDEO and BRIAN2_OPENCV_CAMERA")

if configured_camera is not None:
    try:
        camera_index = int(configured_camera)
    except ValueError as ex:
        raise ValueError("BRIAN2_OPENCV_CAMERA has to be an integer") from ex
    camera_mode = True
    filename = None
    video_source = camera_index
    source_description = f"camera {camera_index}"
else:
    camera_mode = False
    filename = os.path.abspath(
        configured_video or "VID00003-20100701-2204.avi")
    if not os.path.exists(filename):
        if configured_video:
            raise FileNotFoundError(
                f"BRIAN2_OPENCV_VIDEO does not exist: {filename}")
        print('Downloading the example video file')
        response = urllib.request.urlopen(
            "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/"
            "testdata/highgui/video/VID00003-20100701-2204.avi")
        data = response.read()
        with open(filename, 'wb') as f:
            f.write(data)
    video_source = filename
    source_description = filename

video = cv2.VideoCapture(video_source)
if not video.isOpened():
    raise RuntimeError(f"OpenCV could not open {source_description}")
source_width, source_height = (
    int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
if source_width <= 0 or source_height <= 0:
    raise RuntimeError(
        f"OpenCV reported invalid dimensions for {source_description}")
width, height = source_width, source_height
configured_size = os.environ.get("BRIAN2_OPENCV_SIZE")
if configured_size:
    try:
        width_text, height_text = configured_size.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as ex:
        raise ValueError(
            "BRIAN2_OPENCV_SIZE has to use WIDTHxHEIGHT, e.g. 64x36") from ex
    if width <= 0 or height <= 0:
        raise ValueError("BRIAN2_OPENCV_SIZE dimensions have to be positive")

configured_fps = os.environ.get("BRIAN2_OPENCV_FPS")
fps = float(configured_fps) if configured_fps is not None else video.get(
    cv2.CAP_PROP_FPS)
if not np.isfinite(fps) or fps <= 0:
    fps = 24

if configured_frames is not None:
    try:
        requested_frames = int(configured_frames)
    except ValueError as ex:
        raise ValueError("BRIAN2_OPENCV_FRAMES has to be an integer") from ex
    if requested_frames <= 0:
        raise ValueError("BRIAN2_OPENCV_FRAMES has to be positive")
else:
    requested_frames = None

if camera_mode:
    # Cameras are open-ended streams. Keep the example bounded so it is also
    # useful in automated runs and does not need a Python stop callback.
    frame_count = requested_frames or 120
else:
    reported_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_frame_count <= 0:
        raise RuntimeError(
            f"OpenCV reported an invalid frame count for {filename}")
    # Some containers report the number of indexed frames instead of the number
    # the installed decoder can actually return. Count decodable frames so that
    # neither execution path reads past EOF.
    frame_count = 0
    while video.grab():
        frame_count += 1
    video.release()
    if frame_count <= 0:
        raise RuntimeError(f"OpenCV could not decode any frames from {filename}")
    if frame_count != reported_frame_count:
        print(
            f"OpenCV decodes {frame_count} of {reported_frame_count} "
            "reported frames")
    if requested_frames is not None:
        frame_count = min(frame_count, requested_frames)
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        raise RuntimeError(f"OpenCV could not reopen the video: {filename}")

source_size = f"{source_width}x{source_height}"
processing_size = f"{width}x{height}"
size_description = source_size
if processing_size != source_size:
    size_description += f" -> {processing_size}"
print(f"OpenCV source: {source_description}; {size_description} at {fps:g} fps; "
      f"processing {frame_count} frames")
time_between_frames = 1*second/fps
if standalone_device in {"atlas", "rust_standalone"}:
    # NetworkOperation boundaries have to coincide with the model clock. Use
    # the closest representable interval (42 ms for a 24 fps source at 1 ms).
    frame_ticks = max(1, int(round(time_between_frames/defaultclock.dt)))
    time_between_frames = frame_ticks*defaultclock.dt

@implementation('cpp', '''
double* get_frame(bool new_frame)
{
    // The following initializations will only be executed once
    static cv::VideoCapture source(VIDEO_SOURCE);
    static cv::Mat frame;
    static cv::Mat captured_frame;
    static double* grayscale_frame = (double*)malloc(VIDEO_WIDTH*VIDEO_HEIGHT*sizeof(double));
    if (new_frame)
    {
        if (!source.read(captured_frame) || captured_frame.empty())
            throw std::runtime_error("OpenCV could not read the next frame");
        if (captured_frame.cols != VIDEO_WIDTH || captured_frame.rows != VIDEO_HEIGHT)
            cv::resize(captured_frame, frame, cv::Size(VIDEO_WIDTH, VIDEO_HEIGHT));
        else
            frame = captured_frame;
        double mean_value = 0;
        for (int row=0; row<VIDEO_HEIGHT; row++)
            for (int col=0; col<VIDEO_WIDTH; col++)
            {
                const double grayscale_value = (frame.at<cv::Vec3b>(row, col)[0] +
                                                frame.at<cv::Vec3b>(row, col)[1] +
                                                frame.at<cv::Vec3b>(row, col)[2])/(3.0*128);
                mean_value += grayscale_value / (VIDEO_WIDTH * VIDEO_HEIGHT);
                grayscale_frame[row*VIDEO_WIDTH + col] = grayscale_value;
            }
        // subtract the mean
        for (int i=0; i<VIDEO_HEIGHT*VIDEO_WIDTH; i++)
            grayscale_frame[i] -= mean_value;
    }
    return grayscale_frame;
}

double video_input(const int x, const int y)
{
    // Get the current frame (or a new frame in case we are asked for the first
    // element
    double *frame = get_frame(x==0 && y==0);
    return frame[y*VIDEO_WIDTH + x];
}
'''.replace('VIDEO_SOURCE',
            str(camera_index) if camera_mode else json.dumps(filename)),
                libraries=['opencv_core',
                           'opencv_highgui',
                           'opencv_imgproc',
                           'opencv_videoio'],
                headers=['<opencv2/core/core.hpp>',
                         '<opencv2/highgui/highgui.hpp>',
                         '<opencv2/imgproc/imgproc.hpp>',
                         '<stdexcept>'],
                define_macros=[('VIDEO_WIDTH', width),
                               ('VIDEO_HEIGHT', height)])
@check_units(x=1, y=1, result=1)
def video_input(x, y):
    # we assume this will only be called in the custom operation (and not for
    # example in a reset or synaptic statement), so we don't need to do indexing
    # but we can directly return the full result
    success, frame = video.read()
    if not success or frame is None:
        raise RuntimeError(
            f"OpenCV could not read the next frame from {source_description}")
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    grayscale = frame.mean(axis=2)
    grayscale /= 128.  # scale everything between 0 and 2
    return grayscale.ravel() - grayscale.ravel().mean()


N = width * height
tau, tau_th = 10*ms, time_between_frames
G = NeuronGroup(N, '''dv/dt = (-v + I)/tau : 1
                      dv_th/dt = -v_th/tau_th : 1
                      row : integer (constant)
                      column : integer (constant)
                      I : 1 # input current''',
                threshold='v>v_th', reset='v=0; v_th = 3*v_th + 1.0',
                method='exact')
G.v_th = 1
G.row = 'i//width'
G.column = 'i%width'

if standalone_device in {"atlas", "rust_standalone"}:
    @network_operation(dt=time_between_frames, when="start")
    def update_video_input():
        G.I = video_input(0, 0)
else:
    # The generated C++ program opens its own source. In particular, release a
    # camera here so the standalone process can acquire it exclusively.
    video.release()
    G.run_regularly('I = video_input(column, row)',
                    dt=time_between_frames)
mon = SpikeMonitor(G)
runtime = frame_count*time_between_frames
run(runtime, report='text')
if standalone_device in {"atlas", "rust_standalone"}:
    video.release()

# Avoid going through the whole Brian2 indexing machinery too much
i, t, row, column = mon.i[:], mon.t[:], G.row[:], G.column[:]

show_gui = os.environ.get("BRIAN2_OPENCV_GUI", "1").lower() not in {
    "0", "false", "no", "off"
}
if show_gui:
    import matplotlib.animation as animation

    # TODO: Use overlapping windows
    stepsize = 100*ms

    def next_spikes():
        step = 0
        while step*stepsize < runtime:
            spikes = i[(t >= step*stepsize) & (t < (step+1)*stepsize)]
            step += 1
            yield column[spikes], row[spikes]

    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.set_window_title("Brian2 OpenCV spike camera")
    except AttributeError:
        pass
    dots, = ax.plot([], [], 'k.', markersize=2, alpha=.25)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.set_title("space/click: pause  r: restart  q: close")

    def draw_spikes(data):
        x, y = data
        dots.set_data(x, y)
        return dots,

    gui_state = {"paused": False}

    def toggle_pause():
        if gui_state["paused"]:
            ani.resume()
        else:
            ani.pause()
        gui_state["paused"] = not gui_state["paused"]

    def on_key(event):
        if event.key in {" ", "space"}:
            toggle_pause()
        elif event.key == "r":
            ani.frame_seq = ani.new_frame_seq()
            if gui_state["paused"]:
                toggle_pause()
        elif event.key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", lambda event: toggle_pause())
    ani = animation.FuncAnimation(
        fig, draw_spikes, next_spikes, blit=False, repeat=True,
        repeat_delay=1000, cache_frame_data=False)
    plt.show()
