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
        """
        policy (and value) network
        """
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
            """
            reset stage to 0
            clear hidden state
            """
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
        """generate one parameter
        input:
        state: torch.longtensor with size (bs, 1), the sampled action at the last step
        
        return:
        action: torch.longtensor with size (bs, 1)
        log_p_action: torch.floattensor with size (bs, 1)
        value: [optional] torch.floattensor with size (bs, 1)
        """
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
        """generate the whole sequence of parameters
        
        return:
        combo: torch.longtensor with size (bs, space.size(0): 
               (PREVIOUS STATEMENT) num_occlu \times 4 or 7 if color==True)
        log_p_combo: torch.floattensor with size (bs, space.size(0))
        rewards_critic: torch.floatensor with size (bs, space.size(0))
        """
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
        """
        the __init__ function needs to create action space because this relates with 
        the __init__ of the policy network 
        """
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
        # query counter
        self.queries = 0
        
    def create_action_space(self):
        H, W = self.image_tensor.size()[-2:]
        self.H, self.W = H, W
        action_space = []

        # Dynamic per-occlusion shape selection if PA_cfg.patch_shape is None or 'mixed'
        mixed = (getattr(PA_cfg, 'patch_shape', None) is None) or (getattr(PA_cfg, 'patch_shape', None) == 'mixed')

        # compute the base SQUARE side length L from area (A = H*W*area_occlu, L = sqrt(A))
        L = int(torch.tensor([H * W * float(self.area_occlu)]).sqrt_().floor_().long().item())
        L = max(1, min(L, min(H, W)))

        # For equal-area shapes:
        #  - circle radius r_eq = L / sqrt(pi)  (area pi*r^2 == L^2)
        #  - triangle side  l_eq = sqrt(2) * L  (area l^2/2 == L^2)
        r_eq = max(1, int(np.round(L / np.sqrt(np.pi))))
        l_eq = max(1, int(np.round(np.sqrt(2.0) * L)))
        # Bounding box we must reserve for ANY shape (worst case)
        #  - square:   L
        #  - circle:   (2*r_eq + 1)
        #  - triangle: l_eq
        max_dim = max(L, 2 * r_eq + 1, l_eq)

        noise_H = int(self.noises[-1].size(-2))
        noise_W = int(self.noises[-1].size(-1))

        for _ in range(self.num_occlu):
            if mixed:
                # In 'mixed' mode, we choose shape_id per occluder, and (y,x) must accommodate the largest bbox.
                # [shape_id, y, x, noise_id, noise_y, noise_x]
                action_space += [
                    3,                                  # shape_id in {0,1,2} => ['square','circle','triangle']
                    max(1, int(H) - max_dim),           # y (top-left)
                    max(1, int(W) - max_dim),           # x
                    len(self.noises),                   # noise_id
                    max(1, noise_H - max_dim),          # noise_y
                    max(1, noise_W - max_dim),          # noise_x
                ]
            else:
                # Fixed-shape behavior (also equal-area by construction)
                if PA_cfg.patch_shape == 'circle':
                    # radius r = sqrt(A/pi) == L/sqrt(pi)  (equal-area)
                    r = int(np.floor(np.sqrt((H * W * self.area_occlu) / np.pi)))
                    r = max(1, min(r, min(H, W) // 2 - 1))
                    self.size_occlu = r  # radius
                    action_space += [
                        max(1, int(H) - 2 * r),
                        max(1, int(W) - 2 * r),
                        len(self.noises),
                        max(1, noise_H - 2 * r),
                        max(1, noise_W - 2 * r),
                    ]
                elif PA_cfg.patch_shape == 'triangle':
                    # isosceles right triangle with legs l; target area l^2 / 2 = A  => l = sqrt(2A)
                    l = int(np.floor(np.sqrt(2.0 * H * W * self.area_occlu)))
                    l = max(1, min(l, min(H, W)))
                    self.size_occlu = l  # side length
                    action_space += [
                        max(1, int(H) - l),
                        max(1, int(W) - l),
                        len(self.noises),
                        max(1, noise_H - l),
                        max(1, noise_W - l),
                    ]
                else:
                    # square
                    self.size_occlu = L
                    action_space += [
                        max(1, int(H) - L),
                        max(1, int(W) - L),
                        len(self.noises),
                        max(1, noise_H - L),
                        max(1, noise_W - L),
                    ]

        # Save size base-L for training-time rendering in 'mixed'
        if mixed:
            self.size_occlu = L

        return action_space

    def build_robot(self, critic, rl_batch, gamma, lr, stable=True):
        super().__init__(critic, self.action_space, rl_batch, gamma, lr, stable)
        
    def receive_reward(self, p_images, area_occlu, label_batch, target_batch, attack_type, Debug = False):
        """
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
        """
        with torch.no_grad():
            self.model.cuda(torch_cuda)
            self.model.eval()
            p_images, label_batch, target_batch, area_occlu = p_images.cuda(torch_cuda), label_batch.cuda(torch_cuda), target_batch.cuda(torch_cuda), area_occlu.cuda(torch_cuda)
            
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
                

            area_frac = (area_occlu.view(-1, 1).float()) / (self.H * self.W)
            reward_logits = torch.log(p_cl + utils.eps)
            reward = reward_logits - PA_cfg.lambda_area * area_frac
            avg_area = area_frac.mean()

            acc = [label_acc, target_acc]
            filters = [label_filter, target_filter]
            DebugList = [reward_logits, PA_cfg.lambda_area * area_frac, reward, avg_area, (area_occlu.view(-1, 1).float()) ]
            if Debug:
                return reward, acc, avg_area, filters, DebugList, reward_logits
            return reward, acc, avg_area, filters, reward_logits
    
    def reward_backward(self, rewards):
        """
        input:
        reward: torch.floattensor with size (bs, something)
        
        return:
        updated_reward: torch.floattensor with the same size as input
        """
        R = 0
        updated_rewards = torch.zeros(rewards.size()).cuda(torch_cuda)
        for i in range(rewards.size(-1)):
            R = rewards[:, -(i+1)] + self.gamma * R
            updated_rewards[:, -(i+1)] = R
        return updated_rewards
    
    def learn(self, attack_type='target', steps=50, distributed_area=False, base_mask=None):

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
                    p_images, p_masks = self.combo_to_image(
                        combo, self.num_occlu, None,
                        image_batch, self.noises, size_occlu=self.size_occlu,
                        output_p_masks=True                # union mask for each sample
                    )

                    # compute incremental area in pixels (union with base_mask minus base)
                    if base_mask is not None:
                        H, W = self.H, self.W
                        base = base_mask
                        if base.dim() == 3:
                            base = base.squeeze(0)
                        base = base.to(p_masks.device).float()
                        base_area = base.sum().view(1,1).expand(self.rl_batch, 1)
                        base_b = base.unsqueeze(0).expand(self.rl_batch, H, W)
                        union_masks = torch.max(p_masks, base_b)
                        #area_pixels = union_masks.view(self.rl_batch, -1).sum(dim=1, keepdim=True) - base_area
                        area_pixels = union_masks.view(self.rl_batch, -1).sum(dim=1, keepdim=True)

                    else:
                        area_pixels = p_masks.view(self.rl_batch, -1).sum(dim=1, keepdim=True)

                    # receive reward with area penalty
                    r, acc, avg_area, filters, DebugList, reward_logits = self.receive_reward(
                        p_images,
                        area_pixels,
                        label_batch, target_batch,
                        attack_type, Debug= True
                    )
                    rewards[:, -1] = r.squeeze(-1)
                    
                    # orig_r_record
                    orig_r_record.append(reward_logits.mean())
                    
                    # backprop rewards
                    rewards = self.reward_backward(rewards)
                    
                    # update floating variables
                    best_v, best_i = r.max(dim=0)

                    ####################################################################################################
                    # Area debug: build per-shape masks with equal-area formulas from the base-L size
                    if s%20 == 0:
                        H, W = self.H, self.W
                        lam = float(PA_cfg.lambda_area)

                        row = combo[best_i]
                        if row.dim() > 1:
                            row = row.reshape(-1)  # make it 1D: [combo_size]
                        p_l = row.numel() // self.num_occlu

                        per_shape_fracs = []
                        device = row.device

                        # size per occluder (base L used for all shapes)
                        def _L_for(o):
                            so = self.size_occlu
                            if isinstance(so, list):
                                v = so[o]
                                return int(min(v)) if isinstance(v, tuple) else int(v)
                            if isinstance(so, tuple):
                                return int(min(so))
                            return int(so)

                        shp = getattr(PA_cfg, 'patch_shape', None)
                        fixed_sid = None if (shp is None or shp == 'mixed') else (0 if shp=='square' else (1 if shp=='circle' else 2))

                        for o in range(self.num_occlu):
                            base = o * p_l
                            if p_l == 6:
                                sid = int(row[base + 0].item())
                                y0  = int(row[base + 1].item()); x0 = int(row[base + 2].item())
                            else:
                                sid = fixed_sid if fixed_sid is not None else 0
                                y0  = int(row[base + 0].item()); x0 = int(row[base + 1].item())

                            sid = max(0, min(2, sid))
                            L = _L_for(o)

                            m = torch.zeros(H, W, device=device)
                            if sid == 0:  # square (L x L)
                                y1 = max(0, min(H, y0 + L)); x1 = max(0, min(W, x0 + L))
                                if y1 > y0 and x1 > x0:
                                    m[y0:y1, x0:x1] = 1.0
                            elif sid == 1:  # circle, r = round(L / sqrt(pi))
                                r_ = max(1, int(np.round(L / np.sqrt(np.pi))))
                                cy = max(0, min(H - 1, y0 + r_)); cx = max(0, min(W - 1, x0 + r_))
                                yy = torch.arange(H, device=device).view(H, 1)
                                xx = torch.arange(W, device=device).view(1, W)
                                m = (((yy - cy) ** 2 + (xx - cx) ** 2) <= (r_ * r_)).float()
                            else:  # triangle (right isosceles), side = round(sqrt(2)*L)
                                l_t = max(1, int(np.round(np.sqrt(2.0) * L)))
                                l_eff = max(0, min(l_t, H - y0, W - x0))
                                if l_eff > 0:
                                    yy = torch.arange(l_eff, device=device).view(l_eff, 1)
                                    xx = torch.arange(l_eff, device=device).view(1, l_eff)
                                    t = ((yy + xx) <= (l_eff - 1)).float()
                                    m[y0:y0 + l_eff, x0:x0 + l_eff] = torch.max(
                                        m[y0:y0 + l_eff, x0:x0 + l_eff], t
                                    )

                            per_shape_fracs.append(float(m.sum().item() / (H * W)))

                        area_used_frac = float((area_pixels[best_i] / (H * W)).item())
                        lam_area_term = float(DebugList[1][best_i].item())   # = lam * area_used_frac
                        raw_reward    = float(DebugList[0][best_i].item())   # = log(p_cl + eps)
                        pen_reward    = float(DebugList[2][best_i].item())   # = raw - lam*area

                        print(
                            f"[AREA/RWD] step={s} | "
                            f"per-shape(frac)={per_shape_fracs} | "
                            f"union_used(frac)={area_used_frac:.6f} | "
                            f"lambda*area={lam_area_term:.6f} | "
                            f"reward_raw={raw_reward:.6f} | "
                            f"reward_penalized={pen_reward:.6f}"
                        )
                    ####################################################################################################
                    
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
        """
        Render occlusions according to 'combo'.
        IMPORTANT: In 'mixed' mode, 'size_occlu' is the base square side L for every occluder.
                   We derive equal-area parameters per shape inside this function:
                     - square:   side = L
                     - circle:   radius = round(L / sqrt(pi))
                     - triangle: side  = round(sqrt(2) * L)   (right isosceles, area = L^2)
        """
        # --- normalize size_occlu into per-occluder lists (FIXED INIT) ---
        size_list = None
        size_h_list = None
        size_w_list = None

        if isinstance(size_occlu, int):
            size_list = [int(size_occlu)] * num_occlu
        elif isinstance(size_occlu, tuple):
            size_h_list = [int(size_occlu[0])] * num_occlu
            size_w_list = [int(size_occlu[1])] * num_occlu
        elif isinstance(size_occlu, list) and size_occlu and isinstance(size_occlu[0], tuple):
            size_h_list = [int(t[0]) for t in size_occlu]
            size_w_list = [int(t[1]) for t in size_occlu]
        else:
            # list of ints (most common)
            size_list = [int(x) for x in size_occlu]

        bs, combo_size = combo.size()
        p_l = combo_size // num_occlu
        temp_images = image_batch.clone()
        H, W = image_batch.size()[-2:]

        # detect per-occluder shape_id
        has_shape = (p_l == 6)  # [shape_id, y, x, noise_id, noise_y, noise_x]

        if has_shape:
            # slice fields
            p_combo_shape, p_combo_image, p_combo_choose, p_combo_noise = [], [], [], []
            for o in range(num_occlu):
                base = o * p_l
                p_combo_shape.append(combo[:, base + 0 : base + 1])
                p_combo_image.append(combo[:, base + 1 : base + 3])  # y, x
                p_combo_choose.append(combo[:, base + 3 : base + 4]) # noise_id
                p_combo_noise.append(combo[:, base + 4 : base + 6])  # noise_y, noise_x
            p_combo_shape = torch.cat(p_combo_shape, dim=1)
            p_combo_image = torch.cat(p_combo_image, dim=1)
            p_combo_choose = torch.cat(p_combo_choose, dim=1)
            p_combo_noise = torch.cat(p_combo_noise, dim=1)

            # helpers
            def circle_mask_full(H, W, cy, cx, r, device):
                yy = torch.arange(H, device=device).view(H, 1)
                xx = torch.arange(W, device=device).view(1, W)
                return (((yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)).float()

            def triangle_local_mask(l, device):
                yy = torch.arange(l, device=device).view(l, 1)
                xx = torch.arange(l, device=device).view(1, l)
                return ((yy + xx) <= (l - 1)).float()

            # --------- MASK-ONLY PATH (used by calculate_area) ----------
            if noises is None:
                p_masks = torch.zeros(bs, H, W)
                for item in range(bs):
                    for o in range(num_occlu):
                        sid = int(p_combo_shape[item, o].item())
                        sid = max(0, min(2, sid))
                        # base L from size_list if provided; else min(h,w) for rectangular sizes
                        if size_h_list is None:
                            L = int(size_list[o])
                        else:
                            L = int(min(size_h_list[o], size_w_list[o]))

                        y0 = int(p_combo_image[item, 2*o + 0].item())
                        x0 = int(p_combo_image[item, 2*o + 1].item())

                        if sid == 0:  # square (L x L)
                            p_masks[item, y0:y0+L, x0:x0+L] = 1.
                        elif sid == 1:  # circle, r = round(L / sqrt(pi))
                            r = max(1, int(np.round(L / np.sqrt(np.pi))))
                            cy, cx = y0 + r, x0 + r
                            circ = circle_mask_full(H, W, cy, cx, r, p_masks.device)
                            p_masks[item] = torch.max(p_masks[item], circ)
                        else:  # triangle, side = round(sqrt(2)*L)
                            l_t = max(1, int(np.round(np.sqrt(2.0) * L)))
                            l_eff = max(0, min(l_t, H - y0, W - x0))
                            if l_eff <= 0:
                                continue
                            tmask = triangle_local_mask(l_eff, p_masks.device)
                            sl = p_masks[item, y0:y0+l_eff, x0:x0+l_eff]
                            p_masks[item, y0:y0+l_eff, x0:x0+l_eff] = torch.max(sl, tmask)
                return p_masks

            # --------- PASTE TEXTURES (equal-area shapes) ----------
            if output_p_masks:
                p_masks = torch.zeros(bs, H, W)

            for item in range(bs):
                for o in range(num_occlu):
                    sid = int(p_combo_shape[item, o].item())
                    sid = max(0, min(2, sid))

                    nid = int(p_combo_choose[item, o].item())
                    if size_h_list is None:
                        L = int(size_list[o])
                    else:
                        L = int(min(size_h_list[o], size_w_list[o]))

                    y0 = int(p_combo_image[item, 2*o + 0].item())
                    x0 = int(p_combo_image[item, 2*o + 1].item())
                    ny0 = int(p_combo_noise[item, 2*o + 0].item())
                    nx0 = int(p_combo_noise[item, 2*o + 1].item())

                    img_h_room = H - y0
                    img_w_room = W - x0
                    noise_h_room = int(noises[nid].shape[-2]) - ny0
                    noise_w_room = int(noises[nid].shape[-1]) - nx0

                    if sid == 0:  # square (L x L)
                        l_eff = min(L, img_h_room, img_w_room, noise_h_room, noise_w_room)
                        if l_eff <= 0:
                            continue
                        img_win  = temp_images[item][:, y0:y0+l_eff, x0:x0+l_eff]
                        noise_sq = noises[nid][:, ny0:ny0+l_eff, nx0:nx0+l_eff].to(img_win.device)
                        img_win.copy_(noise_sq)
                        if output_p_masks:
                            p_masks[item, y0:y0+l_eff, x0:x0+l_eff] = 1.

                    elif sid == 1:  # circle (r = round(L / sqrt(pi)))
                        target_r = max(1, int(np.round(L / np.sqrt(np.pi))))
                        raw_hbox = 2 * target_r + 1
                        hbox = min(raw_hbox, img_h_room, img_w_room, noise_h_room, noise_w_room)
                        if hbox <= 0:
                            continue
                        r = (hbox - 1) // 2  # actual radius that fits
                        img_win  = temp_images[item][:, y0:y0+hbox, x0:x0+hbox]
                        noise_sq = noises[nid][:, ny0:ny0+hbox, nx0:nx0+hbox].to(img_win.device)

                        yy = torch.arange(hbox, device=img_win.device).view(hbox, 1)
                        xx = torch.arange(hbox, device=img_win.device).view(1, hbox)
                        local_mask = (((yy - r)**2 + (xx - r)**2) <= (r*r)).float().unsqueeze(0)

                        img_win.mul_(1 - local_mask)
                        img_win.add_(noise_sq * local_mask)

                        if output_p_masks:
                            cy, cx = y0 + r, x0 + r
                            circ = circle_mask_full(H, W, cy, cx, r, p_masks.device)
                            p_masks[item] = torch.max(p_masks[item], circ)

                    else:  # triangle (right isosceles), side = round(sqrt(2)*L)
                        l_t = max(1, int(np.round(np.sqrt(2.0) * L)))
                        l_eff = min(l_t, img_h_room, img_w_room, noise_h_room, noise_w_room)
                        if l_eff <= 0:
                            continue
                        img_win  = temp_images[item][:, y0:y0+l_eff, x0:x0+l_eff]
                        tmask = triangle_local_mask(l_eff, img_win.device).unsqueeze(0)
                        noise_sq = noises[nid][:, ny0:ny0+l_eff, nx0:nx0+l_eff].to(img_win.device)
                        img_win.mul_(1 - tmask)
                        img_win.add_(noise_sq * tmask)
                        if output_p_masks:
                            sl = p_masks[item, y0:y0+l_eff, x0:x0+l_eff]
                            p_masks[item, y0:y0+l_eff, x0:x0+l_eff] = torch.max(sl, tmask.squeeze(0))

            if output_p_masks:
                return temp_images, p_masks
            else:
                return temp_images

        # -------------------- fixed-shape fallback (explicit single-shape modes) --------------------
        # split combo into image coords, noise id, noise coords (same as original)
        p_combo_image, p_combo_choose, p_combo_noise = [], [], []
        for o in range(num_occlu):
            p_combo_image.append(combo[:, o*p_l+0 : o*p_l+2])
            p_combo_choose.append(torch.index_select(combo, dim=1, index=torch.LongTensor([o*p_l+2]).cuda(torch_cuda)))
            p_combo_noise.append(combo[:, o*p_l+3 : o*p_l+5])
        p_combo_image = torch.cat(p_combo_image, dim=1)
        p_combo_choose = torch.cat(p_combo_choose, dim=1)
        p_combo_noise = torch.cat(p_combo_noise, dim=1)

        H, W = image_batch.size()[-2:]

        # ---- CIRCLE MODE (radius passed in size_list) ----
        if PA_cfg.patch_shape == 'circle':
            def circle_mask_full(H, W, cy, cx, r, device):
                yy = torch.arange(H, device=device).view(H, 1)
                xx = torch.arange(W, device=device).view(1, W)
                return (((yy - cy)**2 + (xx - cx)**2) <= (r*r)).float()

            # If only masks are requested
            if noises is None:
                p_masks = torch.zeros(bs, H, W)
                device_mask = p_masks.device  # keep CPU like original
                for item in range(bs):
                    for o in range(num_occlu):
                        r = size_list[o]
                        cy = int(p_combo_image[item, o*2+0].item()) + r
                        cx = int(p_combo_image[item, o*2+1].item()) + r
                        circ = circle_mask_full(H, W, cy, cx, r, device_mask)
                        p_masks[item] = torch.max(p_masks[item], circ)
                return p_masks

            if output_p_masks:
                p_masks = torch.zeros(bs, H, W)

            for item in range(bs):
                for o in range(num_occlu):
                    r = size_list[o]
                    cy = int(p_combo_image[item, o*2+0].item()) + r
                    cx = int(p_combo_image[item, o*2+1].item()) + r
                    nid = int(p_combo_choose[item, o].item())
                    ncy = int(p_combo_noise[item, o*2+0].item()) + r
                    ncx = int(p_combo_noise[item, o*2+1].item()) + r

                    y0_i, y1_i = cy - r, cy + r + 1
                    x0_i, x1_i = cx - r, cx + r + 1
                    y0_n, y1_n = ncy - r, ncy + r + 1
                    x0_n, x1_n = ncx - r, ncx + r + 1

                    img_win = temp_images[item][:, y0_i:y1_i, x0_i:x1_i]
                    noise_sq = noises[nid][:, y0_n:y1_n, x0_n:x1_n].to(img_win.device)

                    hbox = 2*r + 1
                    yy = torch.arange(hbox, device=img_win.device).view(hbox,1)
                    xx = torch.arange(hbox, device=img_win.device).view(1,hbox)
                    local_mask = (((yy - r)**2 + (xx - r)**2) <= (r*r)).float().unsqueeze(0)

                    img_win.mul_(1 - local_mask)
                    img_win.add_(noise_sq * local_mask)

                    if output_p_masks:
                        circ = circle_mask_full(H, W, cy, cx, r, p_masks.device)
                        p_masks[item] = torch.max(p_masks[item], circ)

            if output_p_masks:
                return temp_images, p_masks
            else:
                return temp_images
       
        # ---- TRIANGLE MODE (side length passed in size_list) ----
        elif PA_cfg.patch_shape == 'triangle':
            def triangle_local_mask(l, device):
                yy = torch.arange(l, device=device).view(l, 1)
                xx = torch.arange(l, device=device).view(1, l)
                return ((yy + xx) <= (l - 1)).float()  # [l,l]

            if noises is None:
                p_masks = torch.zeros(bs, H, W)  # CPU by default
                for item in range(bs):
                    for o in range(num_occlu):
                        l = size_list[o]
                        y0 = int(p_combo_image[item, o*2+0].item())
                        x0 = int(p_combo_image[item, o*2+1].item())
                        sl = p_masks[item, y0:y0+l, x0:x0+l]
                        tmask = triangle_local_mask(l, sl.device)
                        p_masks[item, y0:y0+l, x0:x0+l] = torch.max(sl, tmask)
                return p_masks

            if output_p_masks:
                p_masks = torch.zeros(bs, H, W)

            for item in range(bs):
                for o in range(num_occlu):
                    l = size_list[o]
                    y0 = int(p_combo_image[item, o*2+0].item())
                    x0 = int(p_combo_image[item, o*2+1].item())
                    nid = int(p_combo_choose[item, o].item())
                    ny0 = int(p_combo_noise[item, o*2+0].item())
                    nx0 = int(p_combo_noise[item, o*2+1].item())

                    img_h_room = H - y0
                    img_w_room = W - x0
                    noise_h_room = int(noises[nid].shape[-2]) - ny0
                    noise_w_room = int(noises[nid].shape[-1]) - nx0

                    l_eff = min(l, img_h_room, img_w_room, noise_h_room, noise_w_room)
                    if l_eff <= 0:
                        continue

                    img_win  = temp_images[item][:, y0:y0+l_eff,  x0:x0+l_eff]
                    noise_sq = noises[nid][:, ny0:ny0+l_eff, nx0:nx0+l_eff].to(img_win.device)
                    tmask = triangle_local_mask(l_eff, img_win.device).unsqueeze(0)

                    img_win.mul_(1 - tmask)
                    img_win.add_(noise_sq * tmask)

                    if output_p_masks:
                        sl = p_masks[item, y0:y0+l_eff, x0:x0+l_eff]
                        p_masks[item, y0:y0+l_eff, x0:x0+l_eff] = torch.max(sl, tmask.squeeze(0))

            if output_p_masks:
                return temp_images, p_masks
            else:
                return temp_images

        # ---- SQUARE MODE ----
        elif PA_cfg.patch_shape == 'square':
            if noises is None:
                p_masks = torch.zeros(bs, H, W)
                for item in range(bs):
                    for o in range(num_occlu):
                        size_o = size_list[o]
                        p_masks[item][
                            p_combo_image[item, o*2+0] : p_combo_image[item, o*2+0]+size_o,
                            p_combo_image[item, o*2+1] : p_combo_image[item, o*2+1]+size_o
                        ] = 1.
                return p_masks
            else:
                for item in range(bs):
                    for o in range(num_occlu):
                        size_o = size_list[o]
                        temp_images[item][:,
                            p_combo_image[item, o*2+0] : p_combo_image[item, o*2+0]+size_o,
                            p_combo_image[item, o*2+1] : p_combo_image[item, o*2+1]+size_o
                        ] = noises[p_combo_choose[item, o]][:,
                            p_combo_noise[item, o*2+0] : p_combo_noise[item, o*2+0]+size_o,
                            p_combo_noise[item, o*2+1] : p_combo_noise[item, o*2+1]+size_o
                        ]
                if output_p_masks:
                    p_masks = torch.zeros(bs, H, W)
                    for item in range(bs):
                        for o in range(num_occlu):
                            size_o = size_list[o]
                            p_masks[item][
                                p_combo_image[item, o*2+0] : p_combo_image[item, o*2+0]+size_o,
                                p_combo_image[item, o*2+1] : p_combo_image[item, o*2+1]+size_o
                            ] = 1.
                    return temp_images, p_masks
                else:
                    return temp_images

            
    @staticmethod
    def from_combos_to_images(x, p_combos, model, area_occlu, noises_used):
        """
        (unchanged signature)
        """
        model = model.cuda(torch_cuda).eval()
        H, W = x.size()[-2:]
        p_images = []
        areas = []
        preds = []
        for index in range(len(p_combos)):
            item = p_combos[index]
            temp_combos = torch.cat(item, dim=1)
            
            shape = getattr(PA_cfg, 'patch_shape', None)
            A = H * W * area_occlu
            if (shape is None) or (shape == 'mixed'):
                size_val = int(torch.tensor([A]).sqrt_().floor_().long().item())  # base L for equal-area shapes
            elif shape == 'circle':
                size_val = int(np.floor(np.sqrt(A / np.pi)))                # radius
            elif shape == 'triangle':
                size_val = int(np.floor(np.sqrt(2.0 * A)))                  # side l
            else:
                size_val = int(torch.tensor([A]).sqrt_().floor_().long().item())  # square side

            p_image, p_masks = TPA_agent.combo_to_image(
                combo=temp_combos,
                num_occlu=len(item),
                mask=None,
                image_batch=x[index].unsqueeze(0),
                noises=noises_used,
                size_occlu=int(size_val),            # base L for mixed; radius for circle; side for triangle.
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
        """
        input:
        p_image: torch.floattensor with size (3, 224, 224)
        return:
        """
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
            shape = getattr(PA_cfg, 'patch_shape', None)
            if (shape is None) or (shape == 'mixed'):
                pre_size_occlu = [ int(torch.tensor([H*W*item]).sqrt_().floor_().long().item())
                                for item in area_sched[:n_pre_agents] ]
            elif shape == 'circle':
                pre_size_occlu = [ int(np.floor(np.sqrt((H*W*item)/np.pi))) for item in area_sched[:n_pre_agents] ]
            elif shape == 'triangle':
                pre_size_occlu = [ int(np.floor(np.sqrt(2.0 * H*W * item))) for item in area_sched[:n_pre_agents] ]
            else:
                pre_size_occlu = [ int(torch.tensor([H*W*item]).sqrt_().floor_().long().item())
                                for item in area_sched[:n_pre_agents] ]


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
            actor.build_robot(critic=False, rl_batch=rl_batch, gamma=1, lr=lr, stable=True)
            selected_combo, r = actor.learn(attack_type=attack_type, steps=steps, base_mask=(p_mask if p_mask.dim()==2 else p_mask.squeeze(0)))
            
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
