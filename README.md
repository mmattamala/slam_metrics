# slam_metrics
Some scripts to calculate metrics used in SLAM. It works in a similar way to Jürgen Sturm's scripts but this unifies both Absolute Trajectory Error (ATE) and Relative Pose Error (RPE), as well as SE(3) ATE and Drift Per Distance Travelled (DDT).

To be used with **Python 3**

## Usage
The main file is `evaluate_metrics.py`. To run, you must execute:

```bash
python evaluate_metrics.py ground_truth_file estimation_file
```

By default, it only provides the absolute RMSE in both translation and orientation. Please check the options to enable the computation of other error statistics.

## Options
Other options available are:

`--offset float`: time offset added to the timestamps of the second file (default: 0.0)

`--scale`: scaling factor for the second trajectory (default: 1.0)

`--max_pairs`: maximum number of pose comparisons (default: 10000, set to zero to disable downsampling)

`--max_difference`: maximally allowed time difference for matching entries (default: 0.02)

`--delta`: delta for evaluation (default: 1.0)

`--delta_unit`: unit of delta: `'s'` for seconds, `'m'` for meters, `'rad'` for radians, `'f'` for frames; default: `'s'`

`--fixed_delta`: only consider pose pairs that have a distance of delta delta_unit (e.g., for evaluating the drift per second/meter/radian)

`--verbose`: print all evaluation data (otherwise, only the RMSE absolute will be printed

`--compute_automatic_scale`: `ATE_Horn` computes the absolute scale using the mod by Raul Mur


## Other software usage
This repository uses code from:
* [TUM scripts](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/) by [Jürgen Sturm](http://jsturm.de/wp/)
* [COPE](https://github.com/dinhhuy2109/python-cope/blob/master/COPE/SE3UncertaintyLib.py) by [Huy Nguyen](https://github.com/dinhhuy2109) (which is based on code by [Barfoot and Furgale](http://asrl.utias.utoronto.ca/code/))
* [evaluate_ate_scale](https://github.com/raulmur/evaluate_ate_scale) modified by [Raul Mur](https://github.com/raulmur)

## References
* **ATE, RPE**: J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, _A Benchmark for the Evaluation of RGB-D SLAM Systems_, In Proc. of the International Conference on Intelligent Robot Systems (IROS), 2012.
* **ATE SE(3), DDT**: R. Scona, S. Nobili, Y. Petillot, and M. Fallon, _Direct Visual SLAM Fusing Proprioception for a Humanoid Robot,_ in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), Vancouver, Canada, 2017.
