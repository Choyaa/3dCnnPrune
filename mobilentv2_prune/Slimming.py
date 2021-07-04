import os
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from models import mobilenetv2_3d
from models.mobilenetv2_3d import InvertedResidual, conv_bn
from Block import *


class BasePruner:
    def __init__(self, model, newmodel, args):
        self.model = model
        self.newmodel = newmodel
        #self.optimizer = optimizer
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, threshold=1e-2)
        self.args = args
        self.blocks = []
    def return_model(self):
        return self.newmodel
    def prune(self):
        self.blocks = []
        for midx, (name, module) in enumerate(self.model.named_modules()):
            idx = len(self.blocks)
            if isinstance(module, InvertedResidual):
                self.blocks.append(InverRes(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))
            if isinstance(module, conv_bn):
                self.blocks.append(CB(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))

            if isinstance(module, nn.Linear):
                self.blocks.append(FC(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))
       # print(self.blocks)
    def clone_model(self):
        blockidx = 0
        for name, m0 in self.newmodel.named_modules():
            if type(m0) not in [InvertedResidual, conv_bn, nn.Linear ]:
                continue
            block = self.blocks[blockidx]
            curstatedict = block.statedict
            if blockidx == 0:
                inputmask = torch.arange(block.inputchannel)
            assert name == block.layername
            if isinstance(block, CB):             #CB
                # conv(1weight)->bn(4weight)->relu
                assert len(curstatedict) == (1 + 4)
                block.clone2module(m0, inputmask)
                inputmask = block.prunemask
            if isinstance(block, InverRes):       #InverRes
                # dw->project or expand->dw->project
                assert len(curstatedict) in (10, 15)
                block.clone2module(m0, inputmask)
                inputmask = torch.arange(block.outputchannel)
            if isinstance(block, FC):             #FC
                block.clone2module(m0, inputmask)
            blockidx += 1
            if blockidx > (len(self.blocks) - 1): break

        for name0, m0 in self.newmodel.named_modules():
            if name0 == 'first_conv.0':
                for name1, m1 in self.model.named_modules():
                    if name1 == 'first_conv.0':
                        break
                m0.weight.data = m1.weight.data
                break

        for name0, m0 in self.newmodel.named_modules():
            if name0 == 'first_conv.1':
                for name1, m1 in self.model.named_modules():
                    if name1 == 'first_conv.1':
                        break
                m0.weight.data = m1.weight.data
                m0.bias.data = m1.bias.data
                m0.running_mean.data = m1.running_mean.data
                m0.running_var.data = m1.running_var.data
                break


#？？？？？？分割原理
def css_thresholding(x, OT_DISCARD_PERCENT, per):
    '''
    MIN_SCALING_FACTOR = 1e-18
    x[x < MIN_SCALING_FACTOR] = MIN_SCALING_FACTOR
    x_sorted, _ = torch.sort(x)
    x2 = x_sorted ** 2
    Z = x2.sum()
    energy_loss = 0
    for i in range(x2.shape[0]):
        energy_loss += x2[i]
        if energy_loss / Z > OT_DISCARD_PERCENT:
            break
    th = (x_sorted[i - 1] + x_sorted[i]) / 2 if i > 0 else 0
    '''
    x = x.abs()
    y, i = torch.sort(x)
    z = len(x)
    th_index = int(z * per)
    th = y[th_index]
    print(z ,  th ,'min and max', y[0] , y[-1])
    return th



class SlimmingPrune(BasePruner):
    def __init__(self, model, newmodel  , args):
        super().__init__(model, newmodel, args)   #111111111
        self.pruneratio = args.pruneratio

    def prune(self):
        super().prune()   #执行baseprune，生成oldmodel，提取18层   22222222

        thres_perlayer = {}
        for b in self.blocks:
            if b.bnscale is not None:
                # bns.extend(b.bnscale.tolist())
                if isinstance(b.bnscale, list):
                    #thres_perlayer[b] = 0
                    thres_perlayer[b] = [css_thresholding(scale, OT_DISCARD_PERCENT=1e-2, per = self.pruneratio) for scale in b.bnscale]
                else:
                    thres_perlayer[b] = css_thresholding(b.bnscale, OT_DISCARD_PERCENT=1e-2, per= self.pruneratio)
        pruned_bn = 0
        for b in self.blocks:
            if b.bnscale is None:
                continue
            thre = thres_perlayer[b]
            if isinstance(b, CB):
                mask = b.bnscale.gt(thre)
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b, InverRes):
                if b.numlayer == 3:
                    mask = b.bnscale.gt(thre)
                    pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                    b.prunemask = torch.where(mask == 1)[0]
                    print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
        if isinstance(self.blocks[-1], FC):  # If the last layer is FC
            # FC layer cannot prune output dimension
            pass

        print('Pre-processing Successful!')

        self.clone_model()
        print('slimming pruner done!!')

