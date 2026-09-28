| model | train traj | group | 1-step RMSE | rollout-40 RMSE | rollout-99 RMSE | IoU@40 | raw mass err | projection L1 |
|---|---|---|---|---|---|---|---|---|
| persistence (φ(t+1)=φ(t)) | - | train/simple | 0.0007 | 0.0263 | 0.0370 | 0.879 | 0.00e+00 | 0.00e+00 |
| persistence (φ(t+1)=φ(t)) | - | test/complex | 0.0008 | 0.0295 | 0.0443 | 0.852 | 0.00e+00 | 0.00e+00 |
| FNO·SDF, large | 6 | train/simple | 0.0020 | 0.0275 | 0.0413 | 0.901 | 2.58e-03 | 2.58e-03 |
| FNO·SDF, large | 6 | test/complex | 0.0017 | 0.0322 | 0.0515 | 0.872 | 1.58e-03 | 1.58e-03 |
| FNO·χ, large | 6 | train/simple | 0.0030 | 0.0302 | 0.0431 | 0.899 | 4.59e-03 | 4.59e-03 |
| FNO·χ, large | 6 | test/complex | 0.0028 | 0.0316 | 0.0470 | 0.883 | 4.36e-03 | 4.36e-03 |
| FNO·SDF, large+aug | 9 | train/simple | 0.0070 | 0.0694 | 0.0780 | 0.767 | 1.75e-01 | 1.75e-01 |
| FNO·SDF, large+aug | 9 | test/complex | 0.0066 | 0.0692 | 0.0793 | 0.773 | 1.48e-01 | 1.48e-01 |
