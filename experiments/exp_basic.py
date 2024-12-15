import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, S_Mamba, S2_Mamba,S3_Mamba,S3_Mamba_Modified, \
    Flashformer_M, Flowformer_M, Autoformer, Autoformer_M, Transformer_M, \
    Informer_M, Reformer_M,PatchTST, iTransformer_LG


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
            'PatchTST': PatchTST,

            'Transformer': Transformer,
            'Transformer_M': Transformer_M,

            'Informer': Informer,
            'Informer_M': Informer_M,
            #'Informer_LG': Informer,

            'Reformer': Reformer,
            'Reformer_M': Reformer_M,
            #'Reformer_LG': Informer_G,

            'Flowformer': Flowformer,
            'Flashformer_M': Flashformer_M,
            #'Flowformer_LG': Flowformer,

            'Flashformer': Flashformer,
            'Flowformer_M': Flowformer_M,
            #'Flashformer_LG': Flashformer,

            'Autoformer': Autoformer,
            'Autoformer_M': Autoformer_M,
            #'Autoformer_LG': Autoformer,

            'iTransformer': iTransformer,
            'iTransformer_LG': iTransformer_LG,
            
            'S_Mamba': S_Mamba,
            'S2_Mamba': S2_Mamba,
            'S3_Mamba': S3_Mamba,
            'S3_Mamba_Modified':S3_Mamba_Modified
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError


    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
