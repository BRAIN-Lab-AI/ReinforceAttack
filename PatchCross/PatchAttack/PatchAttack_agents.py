import os
import time
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

# custom packages
import PatchAttack.utils as utils
from PatchAttack.PatchAttack_config import PA_cfg

# global variables
torch_cuda = 0


class robot():
    
    class p_pi(nn.Module):
        '''
        policy (and value) network
        '''
        def __init__(self, space, embedding_size=30, stable=True, v_theta=False):
            super().__init__()
            self.embedding_size = embedding_size
            embedding_space = [224] + space[:-1]
            # create embedding space
            self.embedding_list = nn.ModuleList([nn.Embedding(embedding_space[i], self.embedding_size) 
                                                 for i in range(len(embedding_space))])
            if stable:
                self._stable_first_embedding()
            # create linear heads
            self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)#(batch, seq, features)
            self.linear_list = nn.ModuleList([nn.Linear(self.embedding_size, space[i])
                                              for i in range(len(space))])
            # create v_theta head, actor-critic mode
            self.v_theta = v_theta
            if self.v_theta:
                self.theta = nn.ModuleList([nn.Linear(self.embedding_size, 1)
                                            for i in range(len(space))])
            # set necessary parameters
            self.stage = 0
            self.hidden = None

        def forward(self, x):
            x = self.embedding_list[self.stage](x)
            # extract feature of current state
            x, self.hidden = self.lstm(x, self.hidden) # hidden: hidden state plus cell state
            # get action prob given the current state
            prob = self.linear_list[self.stage](x.view(x.size(0), -1))
            # get state value given the current state
            if self.v_theta:
                value = self.theta[self.stage](x.view(x.size(0), -1))
                return prob, value
            else:
                return prob

        def increment_stage(self):
            self.stage +=1

        def _stable_first_embedding(self):
            target = self.embedding_list[0]
            for param in target.parameters():
                param.requires_grad = False

        def reset(self):
            '''
            reset stage to 0
            clear hidden state
            '''
            self.stage = 0
            self.hidden = None
    
    def __init__(self, critic, space, rl_batch, gamma, lr, 
                 stable=True):
        # policy network
        self.critic = critic
        self.mind = self.p_pi(space, stable=stable, v_theta=critic)
        
        # reward setting
        self.gamma = gamma # back prop rewards
        # optimizer
        self.optimizer = optim.Adam(self.mind.parameters(), lr=lr)
        
        # useful parameters
        self.combo_size = len(space)
        self.rl_batch = rl_batch
        
    def select_action(self, state):
        '''generate one parameter
        input:
        state: torch.longtensor with size (bs, 1), the sampled action at the last step
        
        return:
        action: torch.longtensor with size (bs, 1)
        log_p_action: torch.floattensor with size (bs, 1)
        value: [optional] torch.floattensor with size (bs, 1)
        '''
        if self.critic:
            p_a, value = self.mind(state)
            p_a = F.softmax(p_a, dim=1)
            
            # select action with prob
            dist = Categorical(probs=p_a)
            action = dist.sample()
            log_p_action = dist.log_prob(action)
            
            return action.unsqueeze(-1), log_p_action.unsqueeze(-1), value
        else:
            p_a = F.softmax(self.mind(state), dim=1)
            
            # select action with prob
            dist = Categorical(probs=p_a)
            action = dist.sample()
            log_p_action = dist.log_prob(action)
            
            return action.unsqueeze(-1), log_p_action.unsqueeze(-1)
    
    def select_combo(self):
        '''generate the whole sequence of parameters
        
        return:
        combo: torch.longtensor with size (bs, space.size(0): 
               (PREVIOUS STATEMENT) num_occlu \times 4 or 7 if color==True)
        log_p_combo: torch.floattensor with size (bs, space.size(0))
        rewards_critic: torch.floatensor with size (bs, space.size(0))
        '''
        state = torch.zeros((self.rl_batch, 1)).long().cuda(torch_cuda)
        combo = []
        log_p_combo = []
        if self.critic:
            # plus r_critic
            rewards_critic = []
            for _ in range(self.combo_size):
                action, log_p_action, r_critic = self.select_action(state)
                combo.append(action)
                log_p_combo.append(log_p_action)
                rewards_critic.append(r_critic)
                
                state = action
                self.mind.increment_stage()
            
            combo = torch.cat(combo, dim=1)
            log_p_combo = torch.cat(log_p_combo, dim=1)
            rewards_critic = torch.cat(rewards_critic, dim=1)
            
            return combo, log_p_combo, rewards_critic
        else:
            for _ in range(self.combo_size):
                action, log_p_action = self.select_action(state)
                combo.append(action)
                log_p_combo.append(log_p_action)
                
                state = action
                self.mind.increment_stage()
            
            combo = torch.cat(combo, dim=1)
            log_p_combo = torch.cat(log_p_combo, dim=1)
            
            return combo, log_p_combo




class TPA_agent(robot):
    
    def __init__(self, model, image_tensor, label_tensor, noises, noises_label, num_occlu, area_occlu):
        '''
        the __init__ function needs to create action space because this relates with 
        the __init__ of the policy network 
        '''
        # BUILD ENVIRONMENT
        self.model = model
        self.image_tensor = image_tensor
        self.label_tensor = label_tensor
        self.noises = noises
        self.noises_label = noises_label
        # build action space
        self.num_occlu = num_occlu
        self.area_occlu = area_occlu
        self.action_space = self.create_action_space()
        #print('environment and action space have been determined')
        #print('remember to build robot')
        
        # query counter
        self.queries = 0
        
    def create_action_space(self):
        H, W = self.image_tensor.size()[-2:]
        self.H, self.W = H, W
        action_space = []
        # cross-only: [x1, y1, x2, y2, noise_id] per occlusion
        self.size_occlu = 1  # dummy for legacy call sites; ignored by renderer
        for _ in range(self.num_occlu):
            action_space += [int(W), int(H), int(W), int(H), int(len(self.noises))]
        return action_space

    def build_robot(self, critic, rl_batch, gamma, lr, stable=True):
        super().__init__(critic, self.action_space, rl_batch, gamma, lr, stable)
        #print('robot built!')
        
    def receive_reward(self, p_images, area_occlu, label_batch, target_batch, attack_type):
        '''
        input:
        p_images: torch.floattensor with size (bs, 3, self.H, self.W)
        area_occlu: torch.floattensor with size (bs, 1), torch.Tensor([0])
        label_batch: torch.longtensor with size (bs, 1)
        target_batch: torch.longtensor with size (bs, 1)
        attack_type: 'target' or 'non_target'
        
        return:
        reward: torch.floattensor with size (bs, 1)
        acc: list of accs [label_acc, target_acc]
        avg_area: the average area, scalar [0, 1.]
        filters: list of filters [label_filter, target_filter]
        '''
        with torch.no_grad():
            self.model.cuda(torch_cuda)
            self.model.eval()
            p_images, label_batch, target_batch, area_occlu =\
            p_images.cuda(torch_cuda), label_batch.cuda(torch_cuda),\
            target_batch.cuda(torch_cuda), area_occlu.cuda(torch_cuda)
            
            output_tensor = self.model(p_images)
            output_tensor = F.softmax(output_tensor, dim=1)
            
            label_acc = utils.accuracy(output_tensor, label_batch)
            label_filter = output_tensor.argmax(dim=1) == label_batch.view(-1)
            
            target_acc = None
            target_filter = None
            
            if attack_type == 'target':
                target_acc = utils.accuracy(output_tensor, target_batch)
                target_filter = output_tensor.argmax(dim=1) == target_batch.view(-1)
                p_cl = torch.gather(input=output_tensor, dim=1, index=target_batch)
            elif attack_type == 'non-target':
                p_cl = 1. - torch.gather(input=output_tensor, dim=1, index=label_batch)
                

            reward = torch.log(p_cl+utils.eps)
            avg_area = area_occlu.mean() / (self.H * self.W)

            acc = [label_acc, target_acc]
            filters = [label_filter, target_filter]
            return reward, acc, avg_area, filters
    
    def reward_backward(self, rewards):
        '''
        input:
        reward: torch.floattensor with size (bs, something)
        
        return:
        updated_reward: torch.floattensor with the same size as input
        '''
        R = 0
        updated_rewards = torch.zeros(rewards.size()).cuda(torch_cuda)
        for i in range(rewards.size(-1)):
            R = rewards[:, -(i+1)] + self.gamma * R
            updated_rewards[:, -(i+1)] = R
        return updated_rewards
    
    def learn(self, attack_type='target', steps=50, distributed_area=False):

        agent_idx = getattr(self, "agent_idx")   # default 0 if not set


        # create image batch
        image_batch = self.image_tensor.expand(self.rl_batch, 
                                               self.image_tensor.size(-3), 
                                               self.H, 
                                               self.W).contiguous()
        label_batch = self.label_tensor.expand(self.rl_batch, 1).contiguous()
        target_batch = self.noises_label.expand(self.rl_batch, 1).contiguous()
        
        # set up training env
        self.mind.cuda(torch_cuda)
        self.mind.train()
        self.optimizer.zero_grad()
        
        # set up non-target attack records
        floating_combo = None
        floating_r = torch.Tensor([-1000]).cuda(torch_cuda)
        
        # set up target attack records
        t_floating_combo = None
        t_floating_r = torch.Tensor([-1000]).cuda(torch_cuda)
        
        # set up orig_reward record to early stop
        orig_r_record = []
        
        # start interacting with the env
        if self.critic:
            pass
        else:
            for s in range(steps):
                
                # add queries
                self.queries += self.rl_batch
                
                # select combo
                combo, log_p_combo = self.select_combo()
                # receive rewards
                rewards = torch.zeros(combo.size()).cuda(torch_cuda)
                
                if distributed_area:
                    pass
                else:
                    p_images = self.combo_to_image(
                        combo, self.num_occlu, None,
                        image_batch, self.noises, size_occlu=self.size_occlu
                    )

                    # receive reward
                    r, acc, avg_area, filters = self.receive_reward(
                        p_images, 
                        torch.Tensor([0]), 
                        label_batch, target_batch, 
                        attack_type, 
                    )
                    r_logits = r
                

                    occlu = self.num_occlu
                    p_l = combo.size(1) // occlu  # expected to be 5 for cross-only renderer

                    for o in range(occlu):
                        base = o * p_l
                        x1 = combo[:, base + 0].float()
                        y1 = combo[:, base + 1].float()
                        x2 = combo[:, base + 2].float()
                        y2 = combo[:, base + 3].float()

                        dx = (x2 - x1).abs()
                        dy = (y2 - y1).abs()









                    rewards[:, -1] = r.squeeze(-1)
                    
                    # orig_r_record
                    orig_r_record.append(r_logits.mean())
                    
                    # backprop rewards
                    rewards = self.reward_backward(rewards)
                    
                    # update floating variables
                    best_v, best_i = r.max(dim=0)
                    if floating_r < best_v:
                        floating_r = best_v
                        floating_combo = combo[best_i]
                        
                    # baseline subtraction
                    rewards = (rewards - rewards.mean()) / (rewards.std() + utils.eps)
                    
                    # calculate loss
                    loss = (-log_p_combo * rewards).sum(dim=1).mean()
                    
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # reset mind
                    self.mind.reset()
                    
                    # early stop
                    if s >= 2:
                        if attack_type == 'target':
                            if acc[1][0] != 0:
                                break
                        elif attack_type == 'non-target':
                            if acc[0][0] != 100:
                                break
                        if abs(orig_r_record[-1] + orig_r_record[-3] - 2*orig_r_record[-2]) < PA_cfg.es_bnd:
                            break
                    
            return floating_combo, floating_r
            
    @staticmethod
    def combo_to_image(combo, num_occlu, mask, image_batch, noises,
                    size_occlu=None, output_p_masks=False):
        """Cross-only renderer.
        Expects per-occlusion tokens: [x1, y1, x2, y2, noise_id].
        If noises is None, return pixel mask only.
        size_occlu is ignored (kept for call-site compatibility).
        """
        bs, combo_size = combo.size()
        H, W = image_batch.size()[-2:]
        device = image_batch.device
        assert combo_size % num_occlu == 0, "Invalid combo layout"
        p_l = combo_size // num_occlu
        assert p_l == 5, f"Expected 5 tokens per occlusion, got {p_l}"
        T = max(1, int(getattr(PA_cfg, 'cross_thickness', 1)))
        min_len = int(getattr(PA_cfg, 'cross_min_len', 2))
        def _clamp(v, lo, hi):
            return int(max(lo, min(hi, v)))
        def line_mask(xa, ya, xb, yb, Y, X, thickness):
            dx = float(xb - xa); dy = float(yb - ya)
            denom = dx*dx + dy*dy + 1e-8
            t = ((X - xa)*dx + (Y - ya)*dy) / denom
            t = t.clamp(0.0, 1.0)
            Xc = xa + t*dx; Yc = ya + t*dy
            d2 = (X - Xc)**2 + (Y - Yc)**2
            return d2 <= ((thickness/2.0)**2)
        temp_images = image_batch.clone()
        need_masks = (noises is None) or output_p_masks
        if need_masks:
            p_masks = torch.zeros(bs, H, W, device=device, dtype=torch.float32)
        for item in range(bs):
            for o in range(num_occlu):
                base = o * p_l
                x1 = int(combo[item, base + 0].item()); y1 = int(combo[item, base + 1].item())
                x2 = int(combo[item, base + 2].item()); y2 = int(combo[item, base + 3].item())
                nid = int(combo[item, base + 4].item()) if noises is not None else 0
                x1 = _clamp(x1, 0, W-1); y1 = _clamp(y1, 0, H-1)
                x2 = _clamp(x2, 0, W-1); y2 = _clamp(y2, 0, H-1)
                if x1 == x2 and y1 == y2:
                    x2 = _clamp(x2 + 1, 0, W-1)
                if abs(x2 - x1) + abs(y2 - y1) < min_len:
                    if x2 + min_len < W:
                        x2 = _clamp(x2 + min_len, 0, W-1)
                    elif x2 - min_len >= 0:
                        x2 = _clamp(x2 - min_len, 0, W-1)
                    elif y2 + min_len < H:
                        y2 = _clamp(y2 + min_len, 0, H-1)
                    else:
                        y2 = _clamp(y2 - min_len, 0, H-1)
                xmin = _clamp(min(x1, x2) - T//2, 0, W-1); xmax = _clamp(max(x1, x2) + T//2, 0, W-1)
                ymin = _clamp(min(y1, y2) - T//2, 0, H-1); ymax = _clamp(max(y1, y2) + T//2, 0, H-1)
                w = xmax - xmin + 1; h = ymax - ymin + 1
                Y = torch.arange(ymin, ymax+1, device=device, dtype=torch.float32).unsqueeze(1).expand(h, w)
                X = torch.arange(xmin, xmax+1, device=device, dtype=torch.float32).unsqueeze(0).expand(h, w)
                maskA = line_mask(x1, y1, x2, y2, Y, X, T)
                maskB = line_mask(x1, y2, x2, y1, Y, X, T)
                union_mask = (maskA | maskB)
                if noises is None:
                    p_masks[item, ymin:ymax+1, xmin:xmax+1] = torch.where(
                        union_mask, torch.tensor(1.0, device=device), p_masks[item, ymin:ymax+1, xmin:xmax+1]
                    )
                else:
                    noise = noises[nid % len(noises)].to(device)
                    _, Ht, Wt = noise.shape
                    if getattr(PA_cfg, 'use_texture_wrap', True):
                        oxA = x1 % Wt; oyA = y1 % Ht
                        oxB = x2 % Wt; oyB = y2 % Ht
                        texA = torch.roll(noise, shifts=(-oyA, -oxA), dims=(1, 2))
                        texB = torch.roll(noise, shifts=(-oyB, -oxB), dims=(1, 2))
                        cropA = texA[:, :h, :w]; cropB = texB[:, :h, :w]
                    else:
                        cropA = noise[:, :h, :w]; cropB = cropA
                    sub = temp_images[item][:, ymin:ymax+1, xmin:xmax+1]
                    sub = torch.where(maskA.unsqueeze(0), cropA, sub)
                    sub = torch.where(maskB.unsqueeze(0), cropB, sub)
                    temp_images[item][:, ymin:ymax+1, xmin:xmax+1] = sub
                    if output_p_masks:
                        p_masks[item, ymin:ymax+1, xmin:xmax+1] = torch.where(
                            union_mask, torch.tensor(1.0, device=device), p_masks[item, ymin:ymax+1, xmin:xmax+1]
                        )
        if noises is None:
            return p_masks
        else:
            if output_p_masks:
                return temp_images, p_masks
            else:
                return temp_images

    @staticmethod
    def from_combos_to_images(x, p_combos, model, area_occlu, noises_used):
        '''
        input:
        x: torch.floattensor with size (bs, 3, 224, 224)
        p_combos: combos returned by DC attack, list of length bs
        model: pytorch model
        area_occlu: float, like 0.04
        noises_used: list of torch.floattensor with size (3, 224, 224)
        
        return:
        areas: list of length bs, occlued areas
        preds: list of length bs, predicted labels
        p_images: list of torch.flosttensor with size (3, 224, 224)
        '''
        model = model.cuda(torch_cuda).eval()
        H, W = x.size()[-2:]
        p_images = []
        areas = []
        preds = []
        for index in range(len(p_combos)):
            item = p_combos[index]
            temp_combos = torch.cat(item, dim=1)
            
            
            size_val = ( torch.Tensor([H*W*area_occlu]).sqrt_().floor_().long().item())
            p_image, p_masks = TPA_agent.combo_to_image(
                combo=temp_combos,
                num_occlu=len(item),
                mask=None,
                image_batch=x[index].unsqueeze(0),
                noises=noises_used,
                size_occlu=int(size_val),            
                output_p_masks=True
            )


            areas.append(p_masks.sum() / (H*W) )

            with torch.no_grad():
                output = F.softmax(model(p_image), dim=1)
                preds.append(output.argmax())

            p_images.append(p_image.squeeze(0))
            print('index of x: {}'.format(index))
        preds = torch.stack(preds, dim=0)
        return areas, preds, p_images
    
    @staticmethod
    def DC(model, p_image, label, noises_used, noises_label, 
           area_sched, n_boxes, attack_type, num_occlu=1, lr=0.03, rl_batch=500, 
           steps=80, n_pre_agents=0, to_load=None, load_index=None):
        '''
        input:
        p_image: torch.floattensor with size (3, 224, 224)
        return:
        '''
        # time to start
        attack_begin = time.time()
        
        # divide and conquer, initialization
        H, W = p_image.size()[-2:]
        p_mask = torch.zeros(H, W).bool()
        
        # set up records
        optimal_combos = [[] for _ in range(n_boxes)]
        non_target_success_rcd = [[] for _ in range(n_boxes)]
        target_success_rcd = [[] for _ in range(n_boxes)]
        queries_rcd = [[] for _ in range(n_boxes)]
        time_used_rcd = [[] for _ in range(n_boxes)]
        
        # load
        if n_pre_agents!=0:
            # Cross-only preload guard
            enable_preload = False
            if to_load is not None:
                try:
                    tmp = torch.cat(to_load.combos[load_index], dim=1)
                    n_pre_combos = len(to_load.combos[load_index])
                    if tmp.size(1) % n_pre_combos == 0 and (tmp.size(1)//n_pre_combos) == 5:
                        enable_preload = True
                except Exception:
                    enable_preload = False
            if not enable_preload:
                n_pre_agents = 0; to_load = None
                print('Skipping preload: incompatible saved combos (cross-only).')
            else:
            
                n_pre_combos = len(to_load.combos[load_index])
            
                # load records
                for a_i in range(n_pre_agents-1, n_boxes):
                    optimal_combos[a_i] = copy.deepcopy(to_load.combos[load_index])
                    non_target_success_rcd[a_i].append(copy.deepcopy(to_load.non_target_success[load_index]))
                    target_success_rcd[a_i].append(copy.deepcopy(to_load.target_success[load_index]))
                    queries_rcd[a_i].append(copy.deepcopy(to_load.queries[load_index]))
                    time_used_rcd[a_i].append(copy.deepcopy(to_load.time_used[load_index]))
            
                # print info
                print('*** loaded pre_maximum_agents: {}'.format(n_pre_agents))
            
                # check success
                stop = to_load.target_success[load_index].item() if attack_type == 'target'\
                else to_load.non_target_success[load_index].item()
            
                # apply combos
                pre_size_occlu = [
                    int(torch.Tensor([H * W * item]).sqrt_().floor_().long().item())
                    for item in area_sched[:n_pre_agents]
                ]

                p_image = TPA_agent.combo_to_image(
                    combo=torch.cat(to_load.combos[load_index], dim=1), 
                    num_occlu=n_pre_combos,
                    mask=None, 
                    image_batch=p_image.unsqueeze(0),  
                    noises=noises_used,
                    size_occlu=pre_size_occlu, 
                    output_p_masks=False
                ).squeeze(0)
            
                if stop:
                    # process records
                    for a_i in range(n_pre_agents, n_boxes):
                        non_target_success_rcd[a_i] = non_target_success_rcd[a_i][-1]
                        target_success_rcd[a_i] = target_success_rcd[a_i][-1]
                        queries_rcd[a_i] = queries_rcd[a_i][-1]
                        time_used_rcd[a_i] = time_used_rcd[a_i][-1]
                    
                    # summury
                    print('*** combos taken: {} | non-target success: {} | target success: {} | queries: {} | '
                          .format((n_pre_combos), to_load.non_target_success[load_index].item(), 
                                  to_load.target_success[load_index].item(), 
                                  to_load.queries[load_index], ))
                    
                    return p_image, optimal_combos, non_target_success_rcd, target_success_rcd, time_used_rcd, queries_rcd
            
                else:
                    assert n_pre_combos == n_pre_agents, 'check your loaded records'

                
            
        # time to restart
        attack_begin = time.time()
        
        # attacking loop
        queries = to_load.queries[load_index] if n_pre_agents!=0 else 0
        for box in range(n_pre_agents, n_boxes):
            actor = TPA_agent(
                model=model,
                image_tensor=p_image, 
                label_tensor=label, 
                noises=noises_used, 
                noises_label=noises_label,
                num_occlu=num_occlu, 
                area_occlu=area_sched[box]
            )
            actor.agent_idx = box              
            actor.build_robot(critic=False, rl_batch=rl_batch, gamma=1, lr=lr, stable=True)
            selected_combo, r = actor.learn(attack_type=attack_type, steps=steps)
            
            # get queries
            queries += actor.queries # query counter
            
            # update p_image, p_mask
            p_image, temp_p_mask = actor.combo_to_image(
                combo=selected_combo, 
                num_occlu=num_occlu, 
                mask=None, 
                image_batch=p_image.unsqueeze(0),
                noises=noises_used, 
                size_occlu=actor.size_occlu,
                output_p_masks=True,
            )
            p_image = p_image.squeeze(0)
            p_mask = p_mask | temp_p_mask.unsqueeze(0).bool()
            
            # check pred
            with torch.no_grad():
                output = F.softmax(model(p_image.unsqueeze(0)), dim=1)
                score, pred = output.max(dim=1)
            
            # get success
            non_target_success = pred != label
            target_success = pred == noises_label

            # show info
            print('combos taken: {} | '
                  'pred: {} | pred_confidence: {:.4f} | '
                  'GT confidence: {:.4f} | target_confidence: {:.4f} | '
                  .format(box+1, 
                          pred.item(), 
                          score.item(), 
                          output[0, label].item(), 
                          output[0, noises_label].item(),
                         )
                 )
            
            # update records
            for temp_box in range(box, n_boxes):
                optimal_combos[temp_box].append(selected_combo)
                non_target_success_rcd[temp_box].append(non_target_success)
                target_success_rcd[temp_box].append(target_success)
                queries_rcd[temp_box].append(queries)
                time_used = time.time() - attack_begin
                time_used_rcd[temp_box].append(time_used)
                
            # check success
            stop = target_success if attack_type == 'target' else non_target_success
            if stop:
                break
        
        # process records
        if n_pre_agents == n_boxes:
            a_i = n_pre_agents - 1
            non_target_success_rcd[a_i] = non_target_success_rcd[a_i][-1]
            target_success_rcd[a_i] = target_success_rcd[a_i][-1]
            queries_rcd[a_i] = queries_rcd[a_i][-1]
            time_used_rcd[a_i] = time_used_rcd[a_i][-1]
        else:
            for a_i in range(n_pre_agents, n_boxes):
                non_target_success_rcd[a_i] = non_target_success_rcd[a_i][-1]
                target_success_rcd[a_i] = target_success_rcd[a_i][-1]
                queries_rcd[a_i] = queries_rcd[a_i][-1]
                time_used_rcd[a_i] = time_used_rcd[a_i][-1]
        
        # summury
        print('*** combos taken: {} | non-target success: {} | target success: {} | queries: {} | '
              .format((box+1) if n_pre_agents!=n_boxes else n_pre_agents, non_target_success_rcd[-1].item(),
              target_success_rcd[-1].item(), queries_rcd[-1], ))
        
        return p_image, optimal_combos, non_target_success_rcd, target_success_rcd, time_used_rcd, queries_rcd
    

