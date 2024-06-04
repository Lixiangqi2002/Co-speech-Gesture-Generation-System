# GENEA Numerical Evaluations

Scripts for numerical evaluations for the GENEA Gesture Generation Challenge 2022:
https://genea-workshop.github.io/2022/challenge/

This directory provides the scripts for quantitative evaluation of our gesture generation framework. We currently support the following measures:

- Histogram of Moving Distance (HMD) for velocity and acceleration
- Hellinger distance between histograms

## Obtain the data

Download the 3D coordinates of the GENEA Challenge 2022 systems at https://zenodo.org/record/6973297 .
Create a `data` folder and put challenge system motions there as in `data/UBA`.

## Run

`calc_histogram.py`, `hellinger_distance.py` and `calc_cca.py` support different quantitative measures, described below.

### Histogram of Moving Distance

Histogram of Moving Distance (HMD) shows the velocity/acceleration distribution of gesture motion.

To calculate HMD, you can use `calc_histogram.py`.
You can select the measure to compute by `--measure` or `-m` option (default: velocity).
In addition, this script supports histogram visualization. To enable visualization, use `--visualize` or `-v` option.

```sh
# Compute velocity histogram
python calc_histogram.py -c your_prediction_dir -m velocity -w 0.05  # You can change the bin width of the histogram

# Compute acceleration histogram
python calc_histogram.py -c your_prediction_dir -m acceleration -w 0.05
```

Note: ` calc_histogram.py` computes HMD for both original and predicted gestures. The HMD of the original gestures will be stored in `result/original` by default.

### Hellinger distance

Hellinger distance indicates how close two histograms are to each other.

To calculate Hellinger distance, you can use `hellinger_distance.py` script.
