import abc

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List
import torch.nn.functional as F

from diffusers.models.cross_attention import CrossAttention

# CAG
class CSAGCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def register_attention_control(unet, controller):

    attn_procs = {}
    lora_procs = {}
    cross_att_count = 0

    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            block_id = '' # mid layer has no block id
            attention_id = int(name[len("mid_block.attentions.")])
            transformer_blocks_id = int(name[len("mid_block.attentions.0.transformer_blocks.")])
            proc_name = name[-15:]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            attention_id = int(name[len("up_blocks.0.attentions.")])
            transformer_blocks_id = int(name[len("up_blocks.0.attentions.0.transformer_blocks.")])
            proc_name = name[-15:]
            hidden_size = unet.config.block_out_channels[block_id]
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            attention_id = int(name[len("down_blocks.0.attentions.")])
            transformer_blocks_id = int(name[len("down_blocks.0.attentions.0.transformer_blocks.")])
            proc_name = name[-15:]
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        processor =  CSAGCrossAttnProcessor(attnstore=controller, place_in_unet=place_in_unet)

        if place_in_unet == "mid":
            lora_proc_name = "unet.{}_block{}.attentions[{}].transformer_blocks[{}].{}".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name)
            lora_procs[lora_proc_name] = eval(lora_proc_name)
            exec("del unet.{}_block{}.attentions[{}].transformer_blocks[{}].{}".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name))
            exec("unet.{}_block{}.attentions[{}].transformer_blocks[{}].{} = processor".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name))
        else:
            lora_proc_name = "unet.{}_blocks[{}].attentions[{}].transformer_blocks[{}].{}".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name)
            lora_procs[lora_proc_name] = eval(lora_proc_name)
            exec("del unet.{}_blocks[{}].attentions[{}].transformer_blocks[{}].{}".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name))
            exec("unet.{}_blocks[{}].attentions[{}].transformer_blocks[{}].{} = processor".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name))

    controller.num_att_layers = cross_att_count
    return lora_procs

def unregister_attention_control(unet, lora_attn_procs):

    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            block_id = '' # mid layer has no block id
            attention_id = int(name[len("mid_block.attentions.")])
            transformer_blocks_id = int(name[len("mid_block.attentions.0.transformer_blocks.")])
            proc_name = name[-15:]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            attention_id = int(name[len("up_blocks.0.attentions.")])
            transformer_blocks_id = int(name[len("up_blocks.0.attentions.0.transformer_blocks.")])
            proc_name = name[-15:]
            hidden_size = unet.config.block_out_channels[block_id]
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            attention_id = int(name[len("down_blocks.0.attentions.")])
            transformer_blocks_id = int(name[len("down_blocks.0.attentions.0.transformer_blocks.")])
            proc_name = name[-15:]
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        if place_in_unet == "mid":
            lora_proc_name = "unet.{}_block{}.attentions[{}].transformer_blocks[{}].{}".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name)
            lora_processor = lora_attn_procs[lora_proc_name]
            exec("unet.{}_block{}.attentions[{}].transformer_blocks[{}].{} = lora_processor".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name))
        else:
            lora_proc_name = "unet.{}_blocks[{}].attentions[{}].transformer_blocks[{}].{}".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name)
            lora_processor = lora_attn_procs[lora_proc_name]
            exec("unet.{}_blocks[{}].attentions[{}].transformer_blocks[{}].{} = lora_processor".format(place_in_unet, block_id, attention_id, transformer_blocks_id, proc_name))


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        #if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        if attn.shape[1] <= 64 ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int, 
                        batch_size: int = 1) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    # dict_keys(['down_cross', 'mid_cross', 'up_cross', 'down_self', 'mid_self', 'up_self'])
    num_pixels = res ** 2

    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]: # dict_keys(['down_cross', 'mid_cross', 'up_cross', 'down_self', 'mid_self', 'up_self'])
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = [out[i*40:(i+1)*40,:,:] for i in range(batch_size)] # 5 layers * 8 head number
    out = torch.stack([attn_map.sum(0) / attn_map.shape[0] for attn_map in out]) # bsz * 64 * 64 * 77
    return out

# =============================================================================================================================================
# SAG
# processes and stores attention probabilities
class CrossAttnStoreProcessor:
    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# =============================================================================================================================================

def pred_x0(sample, model_output, timestep, noise_scheduler):

    alpha_prod_t = noise_scheduler.alphas_cumprod[timestep]

    beta_prod_t = 1 - alpha_prod_t
    if noise_scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif noise_scheduler.prediction_type == "sample":
        pred_original_sample = model_output
    elif noise_scheduler.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        # predict V
        model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
            " or `v_prediction`"
        )

    return pred_original_sample

def sag_masking(unet, original_latents, attn_map, map_size, t, eps, noise_scheduler):
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        bh, hw1, hw2 = attn_map.shape # 8 64 64, this is the stacked self-attention map, num of attention head x height x weight

        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2) # 1, 8, 64, 64

        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0

        attn_mask = (
            attn_mask.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )

        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))
        
        # Blur according to the self-attention mask
        degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)
        degraded_latents = noise_scheduler.add_noise(degraded_latents, noise=eps, timesteps=t)

        return degraded_latents

# Gaussian blur
def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


def latents_to_pil(latents, vae):
    from PIL import Image
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def view_images(images,
        num_rows: int = 1,
        offset_ratio: float = 0.02,
        display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                    w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img
# CAG
#def cag_masking(original_latents, attn_map, map_size, t, eps, noise_scheduler, token_indices, vae):
def cag_masking(original_latents, attn_map, map_size, t, eps, noise_scheduler, token_indices):
    b, latent_channel, latent_h, latent_w = original_latents.shape
    
    attn_maps = []
    for i in range(b):
        token_attn_map = torch.cat([attn_map[i][:,:,t_id].unsqueeze(0) for t_id in token_indices[i]]).unsqueeze(0)
        attn_maps.append(token_attn_map)

    #attn_maps = [attn_map[i][:,:,token_indices[i]].reshape(1,-1,64,64) for i in range(b)]
    attn_mask = torch.stack([am.max(1, keepdim=False).values.sum(1, keepdim=False) > am.max(1, keepdim=False).values.sum(1, keepdim=False).mean() for am in attn_maps])

    attn_mask = (
        attn_mask.reshape(b, map_size[0], map_size[1])
        .unsqueeze(1)
        .repeat(1, latent_channel, 1, 1)
        .type(attn_map.dtype)
    )

    attn_mask = F.interpolate(attn_mask, (latent_h, latent_w)) # bsz 4 64 64
    attn_mask = torch.stack([torch.swapaxes(am, 1, 2) for am in attn_mask])

    # Blur according to the self-attention mask
    degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
    degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)
    degraded_latents = noise_scheduler.add_noise(degraded_latents, noise=eps, timesteps=t)

    return degraded_latents


