"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip_models.blip import BlipBase
from torch import nn
from lavis.models.med import XBertEncoder

from lavis.models.vit import VisionTransformerEncoder
import time


@registry.register_model("blip_image_text_matching")
class BlipITM(BlipBase):
    """
    BLIP Image-Text Matching (ITM) model.

    Supported model types:
        - base: fine-tuned BLIP retrieval weights on COCO dataset (Karpathy split).
        - large: fine-tuned BLIP retrieval weights on COCO dataset (Karpathy split).

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_image_text_matching", "base")
        >>> model = load_model("blip_image_text_matching", "large")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_itm_base.yaml",
        "large": "configs/models/blip_itm_large.yaml",
    }

    def __init__(self, image_encoder, text_encoder, embed_dim=256, max_txt_len=200):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.text_encoder = text_encoder

        self.visual_encoder = image_encoder

        self.max_txt_len = 500 #max_txt_len

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.

        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".

        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.

        Examples:
        ```python
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> caption = "a large fountain spewing water into the air"
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_feature_extractor", is_eval=True)
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> text_input = txt_processors["eval"](caption)

            >>> sample = {"image": image, "text_input": [text_input]}

            >>> features_multimodal = model.extract_features(sample)
            >>> features_multimodal.keys()
            odict_keys(['image_embeds', 'multimodal_embeds'])
            >>> features_multimodal.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_multimodal.multimodal_embeds.shape
            torch.Size([1, 12, 768])

            >>> features_text = model.extract_features(sample, mode="text")
            >>> features_text.keys()
            odict_keys(['text_embeds', 'text_features'])
            >>> features_text.text_embeds.shape
            torch.Size([1, 12, 768])
            >>> features_text.text_features.shape
            torch.Size([1, 12, 256])

            >>> features_image = model.extract_features(sample, mode="image")
            >>> features_image.keys()
            odict_keys(['image_embeds', 'image_features'])
            >>> features_image.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_image.image_features.shape
            torch.Size([1, 197, 256])
        ```
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                    image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return image features
            image_embeds = self.visual_encoder.forward_features(image)

            image_features = self.vision_proj(image_embeds)
            image_features = F.normalize(image_features, dim=-1)

        elif mode == "text":
            assert (
                    caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state

            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel features
            image_embeds = self.visual_encoder.forward_features(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            text.input_ids[:, 0] = self.tokenizer.enc_token_id

            output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    def drop_image_patch_with_highest_att(self, args, drop_iter, img0, img_id):
        # For now, drop simultaneously the highest att patch for all the classes of an image
        for drop in range(drop_iter):  # range start from 0 already
            # print("255 drop iter", drop_iter)
            att_loss_path = f"{args.save_path}/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop}/highest_att_save/highest_att_{img_id}_{args.prune_att_head}.json"
            with open(att_loss_path, 'r') as file:
                att_loss_dict = json.load(file)
            # for class_idx, class_name in enumerate(gt_class_name):
            try:
                # highest_att_idx = att_loss_dict[f"{args.max_att_block_num}_{img_id}_{class_name}"]
                highest_att_idx = att_loss_dict[f"halving_{img_id}"]
            except:
                print(f"{args.max_att_block_num}_{img_id}_{class_name}", att_loss_path)
                print("217 highest att sequence of this image do not exist")
                breakpoint()

            # print("220 max_patch", drop_iter, img_id, class_name, max_patchs)
            patch_num = int(int(args.img_size) / 16)
            for max_patch in max_patchs:
                max_x = (max_patch // patch_num) * 16
                max_y = (max_patch % patch_num) * 16
                # print("img[max_x:max_x+16, max_y:max_y+16] before", max_x, max_x + 16, max_y, max_y + 16, img0[:, max_x:max_x + 16, max_y:max_y + 16].sum(), norm_img0[max_x:max_x + 16, max_y:max_y + 16, :].sum(), img0.sum())
                img0[:, max_x:max_x + 16, max_y:max_y + 16] = 0
                # norm_img0[max_x:max_x + 16, max_y:max_y + 16, :] = 0

        return img0 # norm_img0,
    def forward(self, samples, drop_iter,  match_head="itm"):
        # print("59 BlipITM", self.max_txt_len)
        image = samples["image"]
        caption = samples["text_input"]

        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        print("197 image text matching", image_embeds.shape)
        print("198 image text matching", image_atts.shape)
        
        
        text = self.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        if match_head == "itm":
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id  # extra code

            output = self.text_encoder(
                encoder_input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            return itm_output



        elif match_head == "itc":
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sim = image_feat @ text_feat.t()
            return sim
    def itm_rank(self, image_embeds, image_atts, encoder_input_ids, match_head='itm'):
        # breakpoint()
        encoder_input_ids = encoder_input_ids.clone()
        encoder_input_ids = encoder_input_ids[:, 3:]
        text_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id).long()

        if match_head == 'itm':
            # encoder_input_ids = encoder_input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(encoder_input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            # print(output.last_hidden_state.shape)
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            itm_output = F.softmax(itm_output, dim=1)[:,1]
            return itm_output #, mask, token_length

        elif match_head == 'itc':
            encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
            text_output = self.text_encoder(encoder_input_ids, attention_mask=text_attention_mask,
                                            return_dict=True, mode='text')
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

            sim = image_feat @ text_feat.t()
            return sim

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)

        embed_dim = cfg.get("embed_dim", 256)
        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model


def compute_gradcam(model, visual_input, text_input, tokenized_text, block_num=6):
    # 改 一次forward , 看register hook
    model.text_encoder.base_model.base_model.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True
    # print(model.visual_encoder)
    # model.visual_encoder.base_model.base_model.encoder.layer[
    #     block_num
    # ].crossattention.self.save_attention = True

    output = model({"image": visual_input, "text_input": text_input}, match_head="itm")
    loss = output[:, 1].sum()
    # itm_scores = torch.nn.functional.softmax(output, dim=1)
    # print("159 grad cam itm_scores, loss", loss)

    model.zero_grad()
    loss.backward()

    with torch.inference_mode():
        # cams_withembed = model.text_encoder.base_model.base_model.encoder.layer[
        #     block_num
        # ].crossattention.self.get_attention_map_withembed()
        # token_cos_sim_map(cams_withembed)

    # with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        token_length = tokenized_text.attention_mask.sum(dim=-1) - 2
        token_length = token_length.cpu()
        #token_length = 195
        # grads and cams [bsz, num_head, seq_len, image_patch]

        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attn_gradients()
        #([2, 12, 197, 2305])
        cams = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attention_map()
        # print("192 gradcam min max", cams.max(), cams.min())

        # ([2, 12, 197, 2305])
        # print("186 blip image text matching", mask.shape, token_length, grads.shape, cams.shape)

        # assume using vit with 576 num image patch
        # print(185, "blip img txt matching" , model.text_encoder)
        # print("176 cam grad", cams.shape, grads.shape, mask.shape, tokenized_text.attention_mask.size(0), visual_input.size(), mask[:,:,:cams.shape[2],:,:].shape)
        # if block_num>10:
        #     print("179 cam grad value", cams[1,1,5,:], cams.sum(), grads[1,1,5,:], grads.sum())
        # print("179 cam grad value", cams.shape[2], cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 48, 48).shape, grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 48, 48).shape, mask[:,:,:cams.shape[2],:,:].shape) #cams.shape[2] = the longest caption token nums
        # gradcams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24) * grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 24, 24) * mask[:,:,:cams.shape[2],:,:]
        gradcams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 48, 48) * grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 48, 48) * mask[:,:,:cams.shape[2],:,:]
        # print("204 gradcam min max", gradcams.max(), gradcams.min())

        # assert torch.equal(gradcams_simp, gradcams)

        # print("200 blip img text matching", gradcams.shape)
        gradcam_byatthead_list = []
        for i in range(12):
            gradcam_byatthead_list.append(gradcams[:,i,1:,:,:].cpu().detach())
        gradcam_mean = torch.mean(gradcams.detach(),1)[:,1:,:,:].cpu().detach()


            
    return gradcam_mean, output, gradcam_byatthead_list


def compute_gradcam_ensemble(args, model, visual_input, text_input, tokenized_text, drop_iter):
    # 改 一次forward , 看register hook
    for block_num in range(0, 12):
        # 8th layer 1-12 layers so select the 7th starting from 0
        model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.save_attention = True

    tic_grad1 = time.perf_counter()
    output = model({"image": visual_input, "text_input": text_input}, drop_iter, match_head="itm")
    toc_grad1 = time.perf_counter()
    print(
        f"Count X Time: run model {toc_grad1 - tic_grad1:0.4f} seconds")
    loss = output[:, 1].sum()
    # itm_scores = torch.nn.functional.softmax(output, dim=1)
    # print("159 grad cam itm_scores, loss", loss)

    model.zero_grad()
    loss.backward()

    gradcam_blocklist = []
    cam_blocklist = []
    patch_num = int(args.img_size / 16)
    # print("40 patch_num", patch_num)
    # for block_num in range(int(block_num_range.split("-")[0]), int(block_num_range.split("-")[1])+1):
    for block_num in range(0,12):
        #8th layer 1-12 layers so select the 7th starting from 0
        with torch.inference_mode():

            mask = tokenized_text.attention_mask.view(
                tokenized_text.attention_mask.size(0), 1, -1, 1, 1)  # (bsz,1,token_len, 1,1)

            # grads and cams [bsz, num_head, seq_len, image_patch]
            grads = model.text_encoder.base_model.base_model.encoder.layer[
                block_num
            ].crossattention.self.get_attn_gradients()
            cams = model.text_encoder.base_model.base_model.encoder.layer[
                block_num
            ].crossattention.self.get_attention_map()

            gradcams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, patch_num, patch_num) * grads[:, :, :, 1:].clamp(
                0).reshape(visual_input.size(0), 12, -1, patch_num, patch_num) * mask[:, :, :cams.shape[2], :, :]

            gradcams[gradcams<0] = 0
            gradcam_byatthead_list = []
            for i in range(12):
                gradcam_byatthead_list.append(
                    gradcams[:, i, 1:, :, :].cpu().detach())  # 1: on the third position is for deleting the enc token,

            gradcam_blocklist.append(gradcam_byatthead_list)

            # cam_byatthead_list = []
            # for i in range(12):
            #     cam_byatthead_list.append(
            #         cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, patch_num, patch_num)[:, i, 1:, :, :].cpu().detach())  # 1: on the third position is for deleting the enc token,
            #
            # cam_blocklist.append(cam_byatthead_list)
            # gradcam_mean = torch.mean(gradcams.detach(), 1)[:, 1:, :, :].cpu().detach()
            # gradcam_mean = torch.mean(gradcams, 1)[:, 1:, :, :].cpu().detach()
            # gradcam_list = []
            # for ind in range(visual_input.size(0)):
            #     # token_length_ = token_length[ind]
            #     gradcam = gradcams[ind].mean(0).cpu().detach()
            #     print("289 blip_image_text_matching", gradcam.shape)
            #     # [enc token gradcam, average gradcam across token, gradcam for individual token]
            #     gradcam = gradcam[1:, :]

            # gradcam_list.append(gradcam)
            # gradcam_blocklist.append(gradcam_mean)
    # image_gradcam_blocklist.append(gradcam_byatthead_list)

    return gradcam_blocklist, cam_blocklist, output
