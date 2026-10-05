# Third-party notices

## MediaPipe models — Apache License 2.0
`web/models/*.onnx` are converted from Google MediaPipe models
(palm_detection, hand_landmark, face_detection_short_range, face_landmark,
pose_detection, pose_landmark). Copyright Google LLC.
Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
Changes: converted from TFLite to ONNX, unused outputs removed, weights stored as fp16.

## ONNX Runtime Web 1.30.0 — MIT License
`web/vendor/ort/*` — Copyright (c) Microsoft Corporation.
(`ort-wasm-simd-threaded.jsep.wasm` is stored gzip-compressed as `.wasm.gz`; the local server sends it with `Content-Encoding: gzip`.)
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

## qrcode-generator 2.0.4 — MIT License
`web/vendor/qrcode.js` — Copyright (c) 2009 Kazuhiko Arase. (Same MIT terms as above.)
"QR Code" is a registered trademark of DENSO WAVE INCORPORATED.

## Python packages (installed by start.bat)
OpenVINO (Apache-2.0), NumPy (BSD-3-Clause), aiohttp (Apache-2.0), Pillow (MIT-CMU),
cryptography (Apache-2.0 / BSD).
