"""
How the ONNX models in web/models were produced (for reference / reproducibility).
Not needed to run ClayMirror – the converted models are already included.

Source: Google MediaPipe (Apache-2.0) – the TFLite files shipped inside the
`mediapipe` Python wheel (mediapipe/modules/...):
    palm_detection_full / palm_detection_lite, hand_landmark_full / hand_landmark_lite,
    face_detection_short_range, face_landmark, pose_detection, pose_landmark_full

Steps (Python 3.11, tensorflow-cpu 2.15, tf2onnx 1.16, onnx):
  1. pose_detection.tflite stores sparse fp16 weights (DENSIFY ops) that crash
     tf2onnx → densify them with the TFLite interpreter and rewrite the flatbuffer.
  2. tf2onnx --tflite <model> --opset 13
  3. prune unused outputs (segmentation / heatmap / world landmarks) and store
     weights as fp16 + Cast (compute stays fp32 for every runtime).
  4. verify ONNX vs TFLite outputs (max abs diff ~1e-4) and the full pipeline
     against MediaPipe's own results.

    python tools/convert_models.py <dir with .tflite files> <output dir>
"""
import os
import subprocess
import sys


def densify(src, dst):
    import flatbuffers
    import numpy as np
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as S

    buf = open(src, "rb").read()
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(buf, 0))
    sg = m.subgraphs[0]
    inv = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}
    code = lambda op: inv.get(max(m.operatorCodes[op.opcodeIndex].builtinCode, m.operatorCodes[op.opcodeIndex].deprecatedBuiltinCode))
    it = tf.lite.Interpreter(model_path=src, experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], np.zeros(it.get_input_details()[0]["shape"], np.float32))
    it.invoke()
    ops = []
    for op in sg.operators:
        if code(op) == "DENSIFY":
            dense = it.get_tensor(op.outputs[0])
            b = S.BufferT()
            b.data = np.frombuffer(dense.tobytes(), dtype=np.uint8).copy()
            m.buffers.append(b)
            sg.tensors[op.outputs[0]].buffer = len(m.buffers) - 1
            sparse = sg.tensors[op.inputs[0]]
            sparse.sparsity = None
            sparse.buffer = len(m.buffers) - 1
        else:
            ops.append(op)
    sg.operators = ops
    bld = flatbuffers.Builder(1024)
    bld.Finish(m.Pack(bld), file_identifier=b"TFL3")
    open(dst, "wb").write(bytes(bld.Output()))


def main(src_dir, out_dir):
    tmp = os.path.join(out_dir, "_raw")
    os.makedirs(tmp, exist_ok=True)
    names = ["palm_detection_full", "palm_detection_lite", "hand_landmark_full", "hand_landmark_lite",
             "face_detection_short_range", "face_landmark", "pose_detection", "pose_landmark_full"]
    for n in names:
        src = os.path.join(src_dir, n + ".tflite")
        if n == "pose_detection":
            dense = os.path.join(tmp, n + "_dense.tflite")
            densify(src, dense)
            src = dense
        subprocess.check_call([sys.executable, "-m", "tf2onnx.convert", "--tflite", src,
                               "--output", os.path.join(tmp, n + ".onnx"), "--opset", "13"])
    subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "postprocess_onnx.py"), tmp, out_dir])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
