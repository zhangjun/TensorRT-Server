import tensorrt
import numpy as np
import trt_util

logger = tensorrt.Logger(tensorrt.Logger.VERBOSE)
tensorrt.init_libnvinfer_plugins(logger, "")

def format(weights):
    return tensorrt.Weights(np.ascontiguousarray(weights))

def out(layer, i=0):
    return layer.get_output(i)

def build_network(network, input):
    w1 = network.add_constant((1280,512), format(np.random.random(size=(1280,512)).astype('float32')))
    w2 = network.add_constant((1280,512), format(np.random.random(size=(1280,512)).astype('float32')))
    w3 = network.add_constant((1280,512), format(np.random.random(size=(1280,512)).astype('float32')))

    tmp_w1 = network.add_shuffle(out(w1))
    # s1 = network.add_constant([3], format(np.array([1,1280,512]).astype('int32')))
    # shape1 = [network.add_constant([1], format(np.array([1]).astype('int32'))).get_output(0), network.add_shape(out(w1)).get_output(0)]
    ss1 = network.add_shape(out(w1))
    shape1 = [network.add_constant([1], format(np.array([1]).astype('int32'))).get_output(0), network.add_gather(out(ss1), network.add_constant([1], format(np.array([0]).astype('int32'))).get_output(0), 0).get_output(0), network.add_gather(out(ss1), network.add_constant([1], format(np.array([1]).astype('int32'))).get_output(0), 0).get_output(0)]
    s1 = network.add_concatenation(shape1)
    tmp_w1.set_input(1, out(s1))
    tmp_w1.name = "matmul_1"
    # tmp_w1.reshape_dims = (1, 1280, 512)
    tmp_w2 = network.add_shuffle(out(w2))
    # s2 = network.add_constant([3], format(np.array([1,1280,512]).astype('int32')))
    ss2 = network.add_shape(out(w2))
    shape2 = [network.add_constant([1], format(np.array([1]).astype('int32'))).get_output(0), network.add_gather(out(ss2), network.add_constant([1], format(np.array([0]).astype('int32'))).get_output(0), 0).get_output(0), network.add_gather(out(ss2), network.add_constant([1], format(np.array([1]).astype('int32'))).get_output(0), 0).get_output(0)]
    s2 = network.add_concatenation(shape2)
    tmp_w2.set_input(1, out(s2))
    tmp_w2.name = "matmul_2"
    # tmp_w2.reshape_dims = (1, 1280, 512)
    tmp_w3 = network.add_shuffle(out(w3))
    # s3 = network.add_constant([3], format(np.array([1,1280,512]).astype('int32')))
    ss3 = network.add_shape(out(w3))
    shape3 = [network.add_constant([1], format(np.array([1]).astype('int32'))).get_output(0), network.add_gather(out(ss3), network.add_constant([1], format(np.array([0]).astype('int32'))).get_output(0), 0).get_output(0), network.add_gather(out(ss3), network.add_constant([1], format(np.array([1]).astype('int32'))).get_output(0), 0).get_output(0)]
    s3 = network.add_concatenation(shape3)
    tmp_w3.set_input(1, out(s3))
    tmp_w3.name = "matmul_3"
    # tmp_w3.reshape_dims = (1, 1280, 512)

    m1 = network.add_matrix_multiply(input, tensorrt.MatrixOperation.NONE, out(tmp_w1), tensorrt.MatrixOperation.NONE)
    m2 = network.add_matrix_multiply(input, tensorrt.MatrixOperation.NONE, out(tmp_w2), tensorrt.MatrixOperation.NONE)
    m3 = network.add_matrix_multiply(input, tensorrt.MatrixOperation.NONE, out(tmp_w3), tensorrt.MatrixOperation.NONE)

    r1 = network.add_shuffle(out(m1))
    r1.reshape_dims = (0,0,8,64)   # (1,77,8,64)
    r2 = network.add_shuffle(out(m2))
    r2.reshape_dims = (0,0,8,64)
    r3 = network.add_shuffle(out(m3))
    r3.reshape_dims = (0,0,8,64)

    t1 = network.add_shuffle(out(r1))    # (1, 8, 77, 64)
    t1.first_transpose = (0, 2, 1, 3)
    t2 = network.add_shuffle(out(r2))
    t2.first_transpose = (0, 2, 1, 3)    # (1, 8, 77, 64)
    t3 = network.add_shuffle(out(r3))
    t3.first_transpose = (0, 2, 1, 3)    # (1, 8, 64, 77)

    w = network.add_constant([1], format(np.array([0.125], dtype=np.float32)))
    tmp_w = network.add_shuffle(out(w))
    tmp_w.reshape_dims = (1, 1, 1, 1)
    mul = network.add_elementwise(out(t2), out(tmp_w), tensorrt.ElementWiseOperation.PROD)   # (1, 8, 77, 64)

    # m4 = network.add_matrix_multiply(out(mul), tensorrt.MatrixOperation.NONE, out(t3), tensorrt.MatrixOperation.NONE)
    m4 = network.add_matrix_multiply(out(mul), tensorrt.MatrixOperation.NONE, out(t3), tensorrt.MatrixOperation.TRANSPOSE)

    s = network.add_softmax(out(m4))
    s.axes = 1 << 3
    m5 = network.add_matrix_multiply(out(s), tensorrt.MatrixOperation.NONE, out(t1), tensorrt.MatrixOperation.NONE)

    network.mark_output(out(m5))
    return network

def infer():
    engine_file_path = "trt.engine"
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path,
                    "rb") as f, tensorrt.Runtime(logger) as runtime:
            trt_engine = runtime.deserialize_cuda_engine(f.read())
    else:
        trt_engine = build_engine()

    context = trt_engine.create_execution_context()

def main():
    builder = tensorrt.Builder(logger)
    network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, 7 << 30)

    input = network.add_input(name='input', dtype = tensorrt.float32, shape=(1, 77, 1280))
    profile = builder.create_optimization_profile()
    profile.set_shape(input.name, (1, 77, 1280), (1, 77, 1280), (1, 77, 1280))
    config.add_optimization_profile(profile)

    network = build_network(network, input)
    engine = builder.build_engine(network, config)
    print("Completed creating Engine")
    with open("trt.engine", "wb") as f:
        f.write(engine.serialize())

    context = engine.create_execution_context()

    # set input
    input_datas = []
    fake_input = np.ones([1, 77, 1280], dtype=np.float32)
    input_datas.append(fake_input)
    inputs, outputs, bindings, stream = \
        trt_util.allocate_buffers(context, input_datas)

    output = trt_util.do_inference_v2(context, bindings,
                                        inputs, outputs,
                                        stream)

if __name__ == "__main__":
    main()