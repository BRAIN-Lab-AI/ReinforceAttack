import os
import copy
import numpy as np
import torch
from PatchAttack.PatchAttack_config import PA_cfg
from PatchAttack.PatchAttack_agents import TPA_agent
from easydict import EasyDict as edict
import kornia

torch_cuda = 0

class TPA():
    
    def __init__(self, dir_title):
        
        dir_title = os.path.join(dir_title, PA_cfg.t_name)

        # attack dirs
        self.attack_dirs = [os.path.join(
            'PatchAttack_result',
            'TPA',
            dir_title,
            item,
        ) for item in PA_cfg.TPA_attack_dirs]
        
    def attack(self, model, input_tensor, label_tensor, target, input_name='temp', target_in_dict_mapping=None):
        
        # set attack_dirs
        attack_dirs = [os.path.join(item, input_name, 'texture_used_{}'.format(target)) for item in self.attack_dirs]
        for attack_dir in attack_dirs:
            if not os.path.exists(attack_dir):
                os.makedirs(attack_dir)
        
        # load texture images
        noises = []
        
        if target_in_dict_mapping is None:
            target_in_dict = target
        else:
            target_in_dict = target_in_dict_mapping[target][0]

        # check whether the dictionary has been built
        if os.path.exists(
            os.path.join(
                PA_cfg.texture_template_dirs[target_in_dict], 
                'cluster_{}'.format(PA_cfg.n_clusters-1), 
                'iter_{}.pt'.format(PA_cfg.iter_num)
            )
        ):
            print('texture dictionary of label_{} is already built, loading...'.format(
                target_in_dict))
            
            # load noises
            for c_index in range(PA_cfg.n_clusters):

                noise_to_load = torch.load(os.path.join(
                    PA_cfg.texture_template_dirs[target_in_dict],
                    'cluster_{}'.format(c_index), 
                    'iter_{}.pt'.format(PA_cfg.iter_num)
                ))
                noises.append(noise_to_load)
        else:
            assert False, 'texture dictionary of label {} not found, please generate it'.format(target_in_dict)
        
        # print status
        print('target_{} | {} texture images has been prepared!'.format(
            target_in_dict, len(noises)))

        # release memory
        torch.cuda.empty_cache()
        
        # filter textures
        if PA_cfg.f_noise:
            with torch.no_grad():
                textures_used = []
                for noise in noises:
                    input_noise = texture_generator.spatial_repeat(
                        noise.cuda(torch_cuda), PA_cfg.scale
                    )
                    output = F.softmax(model(input_noise.unsqueeze(0)), dim=1)
                    if output.argmax() == target:
                        textures_used.append(input_noise)
            print('after filtering, there are {} texture images to use'.format(len(textures_used)))
        else:
            textures_used = noises
            print('not filtering the texture images')

        # set up records
        t_rcd = edict()
        t_rcd.combos = []
        t_rcd.non_target_success = []
        t_rcd.target_success = []
        t_rcd.time_used = []
        t_rcd.queries = []
        
        # rcd is originally used for storing results of attacking mulitple images,
        # but here, just for one image
        rcd = [copy.deepcopy(t_rcd) for _ in range(PA_cfg.n_agents)]
        
        # load previous work
        n_pre_agents = 0
        to_load = None
        
        if PA_cfg.run_tag != 'new':
            for agent_index in range(PA_cfg.n_agents-1, -1, -1):
                if os.path.exists(os.path.join(attack_dirs[agent_index], 'finished.pt')):
                    n_pre_agents = agent_index + 1
                    to_load = torch.load(os.path.join(attack_dirs[agent_index], 'rcd.pt'))
                    rcd[agent_index] = copy.deepcopy(to_load)
                    break
        load_rcd = (n_pre_agents != 0)
        
        # attack
        target_tensor = torch.LongTensor([target]).cuda(torch_cuda)
        p_image, combos, non_target_success, target_success, time_used, queries = TPA_agent.DC(
            model=model, 
            p_image=input_tensor, 
            label=label_tensor, 
            noises_used=textures_used,
            noises_label=target_tensor,
            area_sched=PA_cfg.area_sched,
            n_boxes=PA_cfg.n_agents, 
            attack_type='target' if PA_cfg.target else 'non-target',
            num_occlu=PA_cfg.n_occlu, 
            lr=PA_cfg.lr, 
            rl_batch=PA_cfg.rl_batch, 
            steps=PA_cfg.steps,
            n_pre_agents=n_pre_agents,
            to_load=to_load if load_rcd else None,
            load_index=0 if load_rcd else None,  # set it to 0 for single input tensor
        )
        
        # update records
        for a_i in range(n_pre_agents, PA_cfg.n_agents):
            rcd[a_i].combos.append(combos[a_i])
            rcd[a_i].non_target_success.append(non_target_success[a_i])
            rcd[a_i].target_success.append(target_success[a_i])
            rcd[a_i].time_used.append(time_used[a_i])
            rcd[a_i].queries.append(queries[a_i])
            # save records
            torch.save(rcd[a_i], os.path.join(attack_dirs[a_i], 'rcd.pt'))
        
        # flag: finished
        for agent_index in range(n_pre_agents, PA_cfg.n_agents):
            torch.save('finished!', os.path.join(
                attack_dirs[agent_index], 'finished.pt'
            ))
            
        return p_image, rcd
                
    @staticmethod
    def calculate_area(p_image, combos):
        H, W = p_image.size()[-2:]
        pre_size_occlu = [
            torch.Tensor([H*W*item]).sqrt_().floor_().long().item()
            for item in PA_cfg.area_sched
        ]

        p_mask = TPA_agent.combo_to_image(
            combo=torch.cat(combos, dim=1),
            num_occlu=len(combos),
            mask=None,
            image_batch=p_image.unsqueeze(0),
            noises=None,
            size_occlu=pre_size_occlu,
            output_p_masks=True
        ).squeeze(0)
        areas = p_mask.float().sum()/(H*W)
        return areas


            