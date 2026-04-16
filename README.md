Commands for flight path simulations 
python run_simulation.py --path straight --duration 10 --save plot_straight.png --no-show
python run_simulation.py --path arc --duration 10 --save plot_arc.png --no-show
python run_simulation.py --path helix --duration 10 --save plot_helix.png --no-show
python run_simulation.py --path approach --duration 10 --save plot_approach.png --no-show
python run_simulation.py --path figure8 --duration 10 --save plot_figure8.png --no-show
python run_simulation.py --path left_right_5m --duration 10 --save plot_left_right_5m.png --no-show
python run_simulation.py --path left_right_5m_side3m --duration 10 --save plot_left_right_5m_side3m.png --no-show
python run_simulation.py --path circle_10m_1m --duration 10 --save plot_circle_10m_1m.png --no-show
python run_simulation.py --path parabolic_flyby --duration 10 --save plot_parabolic_flyby.png --no-show
python run_simulation.py --path organic --duration 10 --save plot_organic.png --no-show
python run_simulation.py --path spiral_cone_in --duration 10 --save plot_spiral_cone_in.png --no-show
python run_simulation.py --path precision_extent --duration 10 --save plot_precision_extent.png --no-show
python run_simulation.py --path flyby_xneg5_y_sine --duration 10 --save plot_flyby_xneg5.png --no-show

i2s emulations 
python demo_i2s_extract.py --samples 4800 --seed 1
python demo_i2s_extract.py --samples 4800 --seed 1 --write-pcm out.pcm24
python demo_i2s_extract.py --samples 4096 --write-pcm out.pcm24 --write-bits-npy out_bits.npy
python demo_i2s_extract.py --fs 48000 --samples 4096 --source 6 2 4 --snr-db 25 --seed 0
python demo_i2s_extract.py --samples 4800 --no-normalize

Broadband part: white noise is band-limited with a high-pass (~80 Hz) and low-pass (~4 kHz) Butterworth filter, then scaled.
Tonal motor part: adds harmonic sinusoids at k * fundamental_hz (default 120 Hz, 8 harmonics), with:
amplitude rolloff via harmonic_decay and /k,
random phase per harmonic,
small frequency jitter (±2 Hz default) to avoid a perfectly static tone.
