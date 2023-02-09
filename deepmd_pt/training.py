import logging
import torch
import os

from typing import Any, Dict

from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.learning_rate import LearningRateExp
from deepmd_pt.loss import EnergyStdLoss
from deepmd_pt.model import EnergyModel

import time


class Trainer(object):

    def __init__(self, config: Dict[str, Any]):
        '''Construct a DeePMD trainer.

        Args:
        - config: The Dict-like configuration with training options.
        '''
        model_params = config['model']
        training_params = config['training']

        # Iteration config
        self.num_steps = training_params['numb_steps']
        self.disp_file = training_params.get('disp_file', 'lcurve.out')
        self.disp_freq = training_params.get('disp_freq', 1000)
        self.save_ckpt = training_params.get('save_ckpt', 'model.ckpt')
        self.save_freq = training_params.get('save_freq', 1000)

        # Data + Model
        my_random.seed(training_params['seed'])
        dataset_params = training_params.pop('training_data')
        self.training_data = DeepmdDataSet(
            systems=dataset_params['systems'],
            batch_size=dataset_params['batch_size'],
            type_map=model_params['type_map']
        )
        self.model = EnergyModel(model_params, self.training_data)
        if os.getenv('USE_CUDA'):
            self.model = self.model.cuda()

        # Learning rate
        lr_params = config.pop('learning_rate')
        assert lr_params.pop('type', 'exp'), 'Only learning rate `exp` is supported!'
        lr_params['stop_steps'] = self.num_steps
        self.lr_exp = LearningRateExp(**lr_params)

        # Loss
        loss_params = config.pop('loss')
        assert loss_params.pop('type', 'ener'), 'Only loss `ener` is supported!'
        loss_params['starter_learning_rate'] = lr_params['start_lr']
        self.loss = EnergyStdLoss(**loss_params)
        if os.getenv('USE_CUDA'):
            self.loss = self.loss.cuda()

    def run(self):
        '''Start the model training.'''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_exp.start_lr)
        fout = open(self.disp_file, 'w')
        logging.info('Start to train %d steps.', self.num_steps)
        t0 = time.time()
        t_data_total = 0
        for step_id in range(self.num_steps):
            bdata = self.training_data.get_batch()
            optimizer.zero_grad()
            cur_lr = self.lr_exp.value(step_id)

            # Prepare inputs
            coord = torch.from_numpy(bdata['coord'])
            atype = torch.from_numpy(bdata['type'])
            natoms = torch.from_numpy(bdata['natoms_vec'])
            box = torch.from_numpy(bdata['box'])
            l_energy = torch.from_numpy(bdata['energy'])
            l_force = torch.from_numpy(bdata['force'])
            if os.getenv('USE_CUDA'):
                coord = coord.cuda()
                atype = atype.cuda()
                l_energy = l_energy.cuda()
                l_force = l_force.cuda()

            # Compute prediction error
            coord.requires_grad_(True)
            
            t_data = time.time()
            p_energy, p_force = self.model(coord, atype, natoms, box)
            loss, rmse_e, rmse_f = self.loss(cur_lr, natoms, p_energy, p_force, l_energy, l_force)
            # loss_val = loss.detach().cpu().numpy().tolist()
            # logging.info('step=%d, lr=%f, loss=%f', step_id, cur_lr, loss_val)

            # Backpropagation
            loss.backward()
            for g in optimizer.param_groups:
                g['lr'] = cur_lr
            optimizer.step()
            t_data_total += time.time() - t_data

            # Log and persist
            if step_id % self.disp_freq == 0:
                rmse_e_val = rmse_e.detach().cpu().numpy().tolist()
                rmse_f_val = rmse_f.detach().cpu().numpy().tolist()
                
                record = 'step=%d, rmse_e=%f, rmse_f=%f, time=%f, time_data=%f\n' % (step_id, rmse_e_val, rmse_f_val, time.time() - t0, t_data_total)
                fout.write(record)
                fout.flush()
                t0 = time.time()
                t_data_total = 0
            if step_id > 0:
                if step_id % self.save_freq == 0:
                    torch.save(self.model.state_dict(), self.save_ckpt)

        fout.close()
        logging.info('Saving model after all steps...')
        torch.save(self.model.state_dict(), self.save_ckpt)
