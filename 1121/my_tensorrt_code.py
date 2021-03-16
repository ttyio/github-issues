from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading
import time
import math

class PerThreadResouce:
    def __init__(self, engine):
        context = engine.create_execution_context()
        self.context = context

        # prepare buffer
        host_inputs  = []
        cuda_inputs  = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # store
        stream = cuda.Stream()
        self.stream  = stream

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings


class TRTInference:
    def __init__(self, trt_engine_path, trt_engine_datatype, batch_size):
        self.cfx = cuda.Device(0).make_context()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        with open(trt_engine_path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        self.engine  = engine

    def alloc_resource(self):
        return PerThreadResouce(self.engine)

    def infer(self, resource, input_img_path):
        threading.Thread.__init__(self)
        self.cfx.push()

        # restore
        stream  = resource.stream
        context = resource.context

        host_inputs = resource.host_inputs
        cuda_inputs = resource.cuda_inputs
        host_outputs = resource.host_outputs
        cuda_outputs = resource.cuda_outputs
        bindings = resource.bindings

        # read image
        image = 1 - (np.asarray(Image.open(input_img_path), dtype=np.float)/255)
        np.copyto(host_inputs[0], image.ravel())

        # inference
        start_time = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        print("execute times "+str(time.time()-start_time))

        # parse output
        output = np.array([math.exp(o) for o in host_outputs[0]])
        output /= sum(output)
        for i in range(len(output)): print("%d: %.2f"%(i,output[i]))

        self.cfx.pop()


    def destory(self):
        self.cfx.pop()
        return
