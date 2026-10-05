"""Prune unused outputs and store weights as fp16 (+Cast) to shrink ONNX models.
Compute stays fp32 so every runtime (OpenVINO NPU/GPU/CPU, ORT-web WASM/WebGPU/WebNN) can run it."""
import onnx, numpy as np, sys, os
from onnx import helper, numpy_helper, TensorProto

KEEP = {
    'palm_detection_full': ['Identity', 'Identity_1'],
    'palm_detection_lite': ['Identity', 'Identity_1'],
    'hand_landmark_full': ['Identity', 'Identity_1', 'Identity_2'],
    'hand_landmark_lite': ['Identity', 'Identity_1', 'Identity_2'],
    'face_detection_short_range': ['regressors', 'classificators'],
    'face_landmark': ['conv2d_21', 'conv2d_31'],
    'pose_detection': ['Identity', 'Identity_1'],
    'pose_landmark_full': ['Identity', 'Identity_1'],
}

def prune(model, keep):
    outs = [o for o in model.graph.output if o.name in keep]
    # topological backward walk
    needed = set(keep); nodes = list(model.graph.node); keep_nodes = []
    for n in reversed(nodes):
        if any(o in needed for o in n.output):
            keep_nodes.append(n); needed.update(i for i in n.input if i)
    keep_nodes.reverse()
    del model.graph.node[:]; model.graph.node.extend(keep_nodes)
    del model.graph.output[:]; model.graph.output.extend(outs)
    inits = [i for i in model.graph.initializer if i.name in needed]
    del model.graph.initializer[:]; model.graph.initializer.extend(inits)
    vi = [v for v in model.graph.value_info if v.name in needed]
    del model.graph.value_info[:]; model.graph.value_info.extend(vi)
    return model

def fp16_weights(model, min_elems=256):
    new_inits = []; cast_nodes = []
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if arr.size >= min_elems:
                h = arr.astype(np.float16)
                if np.isfinite(h).all() and np.abs(h.astype(np.float32) - arr).max() <= 1e-3 * max(1.0, np.abs(arr).max()):
                    name16 = init.name + '__fp16'
                    new_inits.append(numpy_helper.from_array(h, name16))
                    cast_nodes.append(helper.make_node('Cast', [name16], [init.name], to=TensorProto.FLOAT, name=init.name + '__cast'))
                    continue
        new_inits.append(init)
    del model.graph.initializer[:]; model.graph.initializer.extend(new_inits)
    nodes = cast_nodes + list(model.graph.node)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    return model

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for name, keep in KEEP.items():
        p = os.path.join(src, name + '.onnx')
        if not os.path.exists(p): print('skip', name); continue
        m = onnx.load(p)
        m = prune(m, keep)
        m = fp16_weights(m)
        onnx.checker.check_model(m)
        q = os.path.join(dst, name + '.onnx')
        onnx.save(m, q)
        print(f'{name}: {os.path.getsize(p)/1e6:.1f}MB -> {os.path.getsize(q)/1e6:.1f}MB outputs={[o.name for o in m.graph.output]}')
