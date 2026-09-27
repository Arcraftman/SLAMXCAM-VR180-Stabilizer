# Stabilization analysis and validation, 2026-09-27

## Confirmed causes

1. CLI correction velocity still defaulted to 25 deg/s although the job API
   disabled this limit. Fast camera rotation therefore could not be cancelled.
   CLI now defaults to 0 (disabled), matching the API.
2. CSV job/CLI defaults scaled gyro to 0.45 and shifted pose queries by -167 ms.
   Those values are not a universal sensor calibration: scaling reduces measured
   rotation by 55%, and the offset can put compensation out of phase. Defaults
   are now 1.0 and 0 seconds, including GUI controls. Explicit measured values
   remain supported. SLAM motion input already uses 1.0 and synchronized timing.
3. VQF consumed every sample using the median period even across timestamp gaps.
   Missing samples therefore removed elapsed rotation from the estimate. Fusion
   now interpolates sensor values onto a uniform grid for irregular input and
   interpolates fused poses back onto original timestamps. Invalid timestamps
   and nonfinite sensors fail explicitly.

CLI also exposes the implemented orientation-lock mode and correct mode help.

## Verification

Run from the repository with PowerShell:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m pytest -q -rs --disable-warnings
```

Result: 51 passed. An initial offscreen Qt run yielded 50 passed and 1 skipped;
the subsequent run using the normal Qt platform passed all 51 tests.
Vendored PyVQF emits NumPy 2.5 array-shape deprecation warnings.

For 200 Hz constant 1 rad/s rotation about gravity, dropping 50 samples
(250 ms of data) produced a final orientation difference from the complete
timeline of 14.323945 degrees before the fix and 0.000000 degrees afterwards.
The regression threshold is 0.1 degrees. Other tests cover rapid roll, mixed
axis smoothing, shared stereo sampling, rolling shutter, and rendering.

## Limits and remaining risks

No real video/IMU pair is present in this checkout. These results establish
algorithm correctness for controlled inputs, not measured real-footage quality.
Interpolating gaps cannot reconstruct unobserved motion during a long outage.
CSV synchronization still requires a measured offset; zero is a neutral default,
not automatic synchronization. SLAM frame count equality cannot prove correct
pairing with encoded PTS. Near-vertical horizon locking remains singular, and
180-degree output can reveal missing source pixels under strong correction.
Rotational stabilization cannot remove translation, parallax, or motion blur.
The existing 15 ms VQF orientation averaging also warrants real-footage testing
because it filters the measured pose before compensation.
