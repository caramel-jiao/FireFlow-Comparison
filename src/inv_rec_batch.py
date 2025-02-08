import os
from pathlib import Path
import json
from collections import OrderedDict
import re
import time
from dataclasses import dataclass
from glob import glob
import argparse
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image
from tqdm import tqdm

from flux.sampling import denoise_midpoint, denoise_fireflow, denoise_rf_solver, denoise, denoise_momentum, denoise_momentum_rsp, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5, save_velocity_distribution)
from transformers import pipeline
from PIL import Image
import numpy as np

import os

NSFW_THRESHOLD = 0.85

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)
    
@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    offload: bool = False,
    add_sampling_metadata: bool = True,
):
    """
    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """

    torch_device = torch.device(device)
    name = args.name
    offload = args.offload
    
    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    guidance = args.guidance
    output_dir = args.output_dir
    num_steps = args.num_steps
    prefix = args.output_prefix
    inject = 0
    start_layer_index = args.start_layer_index
    end_layer_index = args.end_layer_index
    seed = args.seed if args.seed > 0 else None

    prefix += '_inject_' + str(inject)
    prefix += '_start_layer_index_' + str(start_layer_index)
    prefix += '_end_layer_index_' + str(end_layer_index)

    base_folder = os.path.join(
        output_dir, 
        args.sampling_strategy + '_step%d' % num_steps + '_cfg%f' % guidance + '_' + prefix
    )
    os.makedirs(base_folder, exist_ok=True)
    
    torch.set_grad_enabled(False)
    img_path_list = sorted(glob(os.path.join(args.img_path, '*')))
    for img_path in tqdm(img_path_list):
        img_name = Path(img_path).name.split('.')[0]

        with open(os.path.join(args.txt_path, img_name + '.txt'), 'r') as f:
            prompt = f.read().replace('\n', '')

        source_prompt = prompt
        target_prompt = prompt

        if name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {name}, chose from {available}")

        if num_steps is None:
            num_steps = 4 if name == "flux-schnell" else 25

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.encoder.to(torch_device)
    
        init_image = None
        init_image_array = np.array(Image.open(img_path).convert('RGB'))
        shape = init_image_array.shape

        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

        init_image = init_image_array[:new_h, :new_w, :]
        width, height = init_image.shape[0], init_image.shape[1]
        
        t0 = time.perf_counter()
        
        init_image = encode(init_image, torch_device, ae)

        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        while opts is not None:
            if opts.seed is None:
                opts.seed = rng.seed()

            opts.seed = None
            if offload:
                ae = ae.cpu()
                torch.cuda.empty_cache()
                t5, clip = t5.to(torch_device), clip.to(torch_device)

            info = {}
            info['feature_path'] = args.feature_path
            info['feature'] = {}
            info['inject_step'] = inject
            info['start_layer_index'] = start_layer_index
            info['end_layer_index'] = end_layer_index
            info['reuse_v']= args.reuse_v
            info['editing_strategy']= args.editing_strategy
            info['qkv_ratio'] = list(map(float, args.qkv_ratio.split(',')))
            
            if not os.path.exists(args.feature_path):
                os.mkdir(args.feature_path)

            inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

            # offload TEs to CPU, load model to gpu
            if offload:
                t5, clip = t5.cpu(), clip.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)
            
            denoise_strategies = {
                'reflow' : denoise,
                'rf_solver' : denoise_rf_solver,
                'fireflow' : denoise_fireflow,
                'rf_midpoint' : denoise_midpoint,
                'momentum_rsp': denoise_momentum_rsp,
                'momentum': denoise_momentum,
            }
            if args.sampling_strategy not in denoise_strategies:
                raise ExceptionType("Unknown denoising strategy")
            denoise_strategy = denoise_strategies[args.sampling_strategy]

            # inversion initial noise
            z, info = denoise_strategy(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
            inp_target["img"] = z

            timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

            # denoise initial noise
            x, _ = denoise_strategy(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)
            
            if offload:
                model.cpu()
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            # decode latents to pixel space
            batch_x = unpack(x.float(), opts.width, opts.height)

            for x in batch_x:
                x = x.unsqueeze(0)

                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                # bring into PIL format and save
                x = x.clamp(-1, 1)
                x = embed_watermark(x.float())
                x = rearrange(x[0], "c h w -> h w c")

                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                
                img.save(os.path.join(base_folder, img_name + '.png'))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RF-Edit')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--guidance', type=float, default=1,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=25,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--start_layer_index', type=int, default=20,
                        help='the number of block which starts to apply the feature sharing')
    parser.add_argument('--end_layer_index', type=int, default=37,
                        help='the number of block which ends to apply the feature sharing')
    parser.add_argument('--output_dir', default='output_inv_rec', type=str,
                        help='the path of the edited image')
    parser.add_argument('--output_prefix', default='inv_rec', type=str,
                        help='prefix name of the edited image')
    parser.add_argument('--sampling_strategy', default='rf_solver', type=str,
                        help='method used to conduct sampling at inference time')
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--reuse_v', type=int, default=1,
                        help='reuse v during inversion and reconstruction/editing')
    parser.add_argument('--editing_strategy', default='replace_v', type=str,
                        help='strategy for editing')
    parser.add_argument('--qkv_ratio', type=str, default='1.0,1.0,1.0', help='A string of comma-separated float numbers')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    
    parser.add_argument('--img_path', default='/home/jovyan/shared/jiaoguanlong/shared-datasets/cc3m/validation', type=str,
                        help='root of evaluation imags (cc3m)')
    parser.add_argument('--txt_path', default='/home/jovyan/shared/jiaoguanlong/shared-datasets/cc3m/validation_txt', type=str,
                        help='root of evaluation texts (cc3m)')
    
    args = parser.parse_args()
    
    main(args)