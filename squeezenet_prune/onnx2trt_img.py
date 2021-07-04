import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os
import time
import torchvision
from model import generate_model
from opts import parse_opts
from spatial_transforms import *
from mean import get_mean, get_std
from target_transforms import ClassLabel, VideoID
from dataset import get_training_set, get_validation_set, get_test_set

max_batch_size = 1
onnx_model_path = 'squeezenet.onnx'
trt_engine_path = 'squeezenet.trt'

TRT_LOGGER = trt.Logger()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    
    def __str__(self):
        return "Host: \n" + str(self.host) + '\nDevice:\n' + str(self.device)
    
    def __repr__(self):
        return self.__str__()
    

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", save_engine=False):

    def build_engine(max_batch_size, save_engine):
        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = max_batch_size
            
            builder.fp16_mode = False  
            builder.int8_mode = False 
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
            last_layer = network.get_layer(network.num_layers - 1)
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_cuda_engine(network)
            print('Completed creating Engine')

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine
    if os.path.exists(engine_file_path):
        print('Reading engine from file {}'.format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)
    
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

def main():
    opt = parse_opts()
    img = np.ones([1,2,8,112,112])*0.5
    img = img.astype(dtype = np.float32)
    # print(type(img))
    # print(type(img[0]))
##############



    fp16_mode = False
    int8_mode = False
    engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, save_engine=True)

    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)   

    shape_of_output = (max_batch_size, 13)
    ##
    inputs[0].host = img.reshape(-1)
    print(type(inputs))

    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    t2 = time.time()
    output_trt = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    print('trt:out',output_trt)

    print('TensorRT OK!')
    print(np.argmax(output_trt))
###################pytorch######################
    
    
    opt.arch = '{}'.format(opt.model)
    model, parameters = generate_model(opt)
    model = model.eval()

    input_for_torch = torch.from_numpy(img).cuda()
    t3 = time.time()
    output_torch = model(input_for_torch)
    t4 = time.time()
    print('pytorch out:',output_torch)
    output_torch = output_torch.cpu().data.numpy()
    print('Pytorch OK!')
    # print(np.argmax(output_torch))

    mae = np.mean(abs(output_trt - output_torch))
    print('Inference time with the TensorRT engine: {}'.format(t2-t1))
    print('Inference time with the Pytorch model: {}'.format(t4-t3))
    print('MAE = {}'.format(mae))

    print('All completed!')

if __name__ == '__main__':
    main()

