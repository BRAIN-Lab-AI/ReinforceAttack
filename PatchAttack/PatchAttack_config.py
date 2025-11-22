import os
import numpy as np
from easydict import EasyDict as edict


# PatchAttack config
PA_cfg = edict() 

# config PatchAttack
def configure_PA(t_name, t_labels, 
                 target=False, area_occlu=0.035, n_occlu=1, rl_batch=500, steps=50,
                 TPA_n_agents=10,
                 shape = None, lambda_area = 0
                 ):
    """
    Notes on shapes:
      - shape=None or 'mixed' => the agent chooses shape per occluder. We ensure all shapes
        (square, circle, triangle) have the SAME effective area for a given area_occlu by using:
            L = sqrt(A) where A = H*W*area_occlu  (square side)
            circle radius r = L / sqrt(pi)
            triangle side  l = sqrt(2) * L   (right isosceles)
      - shape='circle' => we use r = sqrt(A/pi)
      - shape='triangle' => we use l = sqrt(2A)
      - shape='square' => we use side L = sqrt(A)
    """

    # Dictionry's shared params
    PA_cfg.t_name = t_name
    PA_cfg.t_labels = t_labels

    # Texture dictionary
    PA_cfg.conv = 5
    PA_cfg.style_layer_choice = [1, 6, 11, 20, 29][:PA_cfg.conv]
    PA_cfg.style_channel_dims = [64, 128, 256, 512, 512][:PA_cfg.conv]
    PA_cfg.cam_thred = 0.8
    PA_cfg.n_clusters = 30
    PA_cfg.cls_w = 0
    PA_cfg.scale = 1
    PA_cfg.iter_num = 9999

    # AdvPatch dictionary
    PA_cfg.f_noise = 'VGG19'
    PA_cfg.feature_layer = 'relu_2_1'
    PA_cfg.texture_k_clusters = 30
    PA_cfg.k_sieve_ratio = 0.5
    PA_cfg.alpha_style = 1.1
    PA_cfg.alpha_content = 10.0
    PA_cfg.alpha_tv = 0.001
    PA_cfg.jitter_max = 1.1
    PA_cfg.flip_p = 0.5
    PA_cfg.lr = 0.03
    PA_cfg.es_bnd = 1e-4
    PA_cfg.image_shape = (3, 224, 224)
    PA_cfg.scale_min = 0.9
    PA_cfg.scale_max = 1.1
    PA_cfg.rotate_max = 22.5
    PA_cfg.rotate_min = -22.5
    PA_cfg.batch_size = 16
    PA_cfg.percentage = 0.09
    PA_cfg.AP_lr = 10.0
    PA_cfg.iterations = 500


    # Attack's shared params
    PA_cfg.target = target
    PA_cfg.area_occlu = area_occlu
    PA_cfg.n_occlu = n_occlu
    PA_cfg.rl_batch = rl_batch
    PA_cfg.steps = steps
    PA_cfg.n_agents = TPA_n_agents
    if not hasattr(PA_cfg, 'area_sched'):
        PA_cfg.area_sched = []
    if PA_cfg.area_sched == []:
        PA_cfg.area_sched = [PA_cfg.area_occlu] * PA_cfg.n_agents
    # None => per-occlusion (mixed) shapes; otherwise one of {'square','circle','triangle'}
    PA_cfg.patch_shape = shape
    PA_cfg.circle_r_pixel = None
    PA_cfg.lambda_area = float(lambda_area)   # strength of area penalty; 0 disables it

    # TPA
    PA_cfg.n_agents = TPA_n_agents
    PA_cfg.f_noise = False  # filter the textures (default: False)
    PA_cfg.es_bnd = 1e-4    # early stop bound (default: 1e-4)
    PA_cfg.area_occlu = area_occlu
    PA_cfg.area_sched = []
    if PA_cfg.area_sched == []:
        PA_cfg.area_sched = [PA_cfg.area_occlu] * PA_cfg.n_agents
    PA_cfg.patch_shape = shape
    PA_cfg.circle_r_pixel = None
    PA_cfg.lambda_area = float(lambda_area)

    # Texture dict dirs
    texture_dirs = []
    texture_sub_dirs = []
    texture_template_dirs = []

    for t_label in PA_cfg.t_labels:
        texture_dir = os.path.join(
            PA_cfg.t_name,
            'attention-style_t-label_{}'.format(t_label)
        )
        texture_sub_dir = os.path.join(
            texture_dir,
            'conv_{}_cam-thred_{}_n-clusters_{}'.format(
                PA_cfg.conv, PA_cfg.cam_thred, PA_cfg.n_clusters
            )
        )
        texture_template_dir = os.path.join(
            texture_sub_dir,
            'cls-w_{}_scale_{}'.format(
                PA_cfg.cls_w, PA_cfg.scale,
            )
        )
        texture_dirs.append(texture_dir)
        texture_sub_dirs.append(texture_sub_dir)
        texture_template_dirs.append(texture_template_dir)
        
    PA_cfg.texture_dirs = texture_dirs
    PA_cfg.texture_sub_dirs = texture_sub_dirs
    PA_cfg.texture_template_dirs = texture_template_dirs

    # AdvPatch dict dirs
    PA_cfg.AdvPatch_dirs = []
    for t_label in PA_cfg.t_labels:
        AdvPatch_dir = os.path.join(
            PA_cfg.t_name,
            't-label_{}'.format(
                t_label
            ),
            'percentage_{}'.format(
                PA_cfg.percentage,
            ),
            'scale_{}-{}_rotate_{}-{}'.format(
                PA_cfg.scale_min, PA_cfg.scale_max,
                PA_cfg.rotate_min, PA_cfg.rotate_max,
            ),
            'LR_{}_batch_size_{}_iterations_{}'.format(
                PA_cfg.AP_lr, PA_cfg.batch_size, PA_cfg.iterations
            ),
        )
        PA_cfg.AdvPatch_dirs.append(AdvPatch_dir)

    # TPA attack dirs
    TPA_attack_dirs = []
    for agent_index in range(PA_cfg.n_agents):
        attack_dir = os.path.join(
            'PatchAttack_result',
            PA_cfg.t_name,
            'TPA',
            'target_{}_occlu_{}_dir_{}_noise_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                int(PA_cfg.target), PA_cfg.n_occlu, 0, PA_cfg.f_noise,
                PA_cfg.feature_layer, PA_cfg.texture_k_clusters, PA_cfg.k_sieve_ratio,
                PA_cfg.alpha_style, PA_cfg.alpha_content, PA_cfg.alpha_tv, 
                PA_cfg.lr, PA_cfg.flip_p
            ),
            'n-occlu_{}_f-noise_{}_lr_{}_rl-batch_{}_steps_{}_es-bnd_{}'
            .format(
                PA_cfg.n_occlu, PA_cfg.f_noise, 
                PA_cfg.lr, PA_cfg.rl_batch, PA_cfg.steps, PA_cfg.es_bnd,
            ),
            'area-sched_'+'-'.join(
                [str(item) for item in PA_cfg.area_sched[:agent_index+1]]
            )+\
            '_n-agents_{}'.format(agent_index+1),
        )
        TPA_attack_dirs.append(attack_dir)
    PA_cfg.TPA_attack_dirs = TPA_attack_dirs
