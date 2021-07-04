import torch
import torchvision
from model import generate_model
from opts import parse_opts

def get_onnx(model, onnx_save_path, example_tensor):
    example_tensor = example_tensor.cuda()
    _ = torch.onnx.export(model, 
            example_tensor,
            onnx_save_path,
            verbose = True,
            training = False,
            do_constant_folding = False,
            input_names = ['input'],
            output_names = ['output'])


if __name__ == '__main__':

    opt = parse_opts()
    opt.arch = '{}'.format(opt.model)
    opt.pretrain_path = r'/home/choya/gitee/squeezenet_pruned/results/SHGD_prune/SHGD_squeezenet_IRD_8_best.pth'
    model,parameters = generate_model(opt)
    onnx_save_path = "squeezenet.onnx"
    example_tensor = torch.randn(1,2, 8,112,112,device = 'cuda')
    get_onnx(model, onnx_save_path, example_tensor)
