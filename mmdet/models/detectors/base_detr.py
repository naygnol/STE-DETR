# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union
import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
import torchvision
from mmdet.models.Add.transformer import build_glimpse_transformer
import copy
import math
import torch.nn.functional as F
from mmdet.models.Add import box_ops
from mmdet.models.layers.transformer.utils import inverse_sigmoid

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
@MODELS.register_module()
class DetectionTransformer(BaseDetector, nn.Module, metaclass=ABCMeta):
    r"""Base class for Detection Transformer.

    In Detection Transformer, an encoder is used to process output features of
    neck, then several queries interact with the encoder features using a
    decoder and do the regression and classification with the bounding box
    head.

    Args:
        backbone (:obj:`ConfigDict` or dict): Config of the backbone.
        neck (:obj:`ConfigDict` or dict, optional): Config of the neck.
            Defaults to None.
        encoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict, optional): Config
            of the positional encoding module. Defaults to None.
        num_queries (int, optional): Number of decoder query in Transformer.
            Defaults to 100.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            the bounding box head module. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            the bounding box head module. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 100,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # process args
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)



        ##cls_branch
        self.use_rego = True
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        glimpse_transformer_cls = build_glimpse_transformer()
        if self.use_rego:
            # LFOM box enlarge ratio λ; paper best is 1.75 on COCO
            self.rego_scales_cls = [1.75]

            self.dropout_cls = nn.Dropout(p=0.01)
            self.roi_query_dim_cls = 256
            self.feat_gp_cls = 4
            self.roi_feat_dim_cls = self.roi_query_dim_cls #* self.feat_gp

            self.ctx_ch_cls = 64
            ctx_inconvs_cls = []
            ctx_outconvs_cls = []
            ctx_gns = []
            hidden_dim = 256
            num_classes = 80
            for i in range(4):
                for gi in range(3):
                    ctx_inconvs_cls.append(nn.Conv2d(256, self.ctx_ch_cls, kernel_size=3, stride=1, padding=(3+gi*4), dilation=(3+gi*4), groups=8))
                    ctx_outconvs_cls.append(nn.Conv2d(self.ctx_ch_cls, 256, kernel_size=1, stride=1, padding=0))
                ctx_gns.append(nn.GroupNorm(32, 256))

            self.ctx_inconvs_cls = nn.ModuleList(ctx_inconvs_cls)
            self.ctx_outconvs_cls = nn.ModuleList(ctx_outconvs_cls)
            self.ctx_gns_cls = nn.ModuleList(ctx_gns)
            for mm in self.ctx_inconvs_cls.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.xavier_normal_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)
            for mm in self.ctx_outconvs_cls.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.normal_(mm.weight, mean=0., std=1e-3)
                    nn.init.constant_(mm.bias, 0.0)

            self.roi_ext_cls = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat2', 'feat3', 'feat4'], 7, 2)
            num_pred = glimpse_transformer_cls.decoder.num_layers
            for gi in range(len(self.rego_scales_cls)):
                rcnn_net_cls = nn.Sequential( *[nn.Conv2d(hidden_dim, self.roi_feat_dim_cls, kernel_size=7, stride=1, padding=0, groups=self.feat_gp_cls),   #
                                            nn.Flatten(1), nn.LayerNorm(self.roi_feat_dim_cls), nn.ReLU(),
                                            nn.Linear(self.roi_feat_dim_cls, self.roi_query_dim_cls), nn.LayerNorm(self.roi_query_dim_cls) ])
                setattr(self, 'rcnn_net_%d_cls'%gi, rcnn_net_cls)
                if gi == 0:
                    setattr(self, 'glimpse_transformer_%d_cls'%gi, glimpse_transformer_cls)
                else:
                    setattr(self, 'glimpse_transformer_%d_cls'%gi, copy.deepcopy(glimpse_transformer_cls))

                rego_hs_linear_cls = nn.Linear((gi+1) * hidden_dim, hidden_dim, bias=False)
                rego_hs_linear_norm_cls = nn.LayerNorm(hidden_dim)
                setattr(self, 'rego_hs_linear_%d_cls'%gi, rego_hs_linear_cls)
                setattr(self, 'rego_hs_linear_norm_%d_cls'%gi, rego_hs_linear_norm_cls)

                rego_hs_fuser_cls = nn.Linear((gi+2) * hidden_dim, hidden_dim, bias=False)
                setattr(self, 'rego_hs_fuser_%d_cls'%gi, _get_clones(rego_hs_fuser_cls, num_pred))
                setattr(self, 'layer_norms_%d_cls'%gi, nn.ModuleList([nn.LayerNorm(hidden_dim) for i in range(num_pred)]))

                rego_class_embed_cls = nn.Linear(hidden_dim, num_classes)
                # rego_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                rego_class_embed_cls.bias.data = torch.ones(num_classes) * bias_value
                # nn.init.constant_(rego_bbox_embed.layers[-1].weight.data, 0)
                # nn.init.constant_(rego_bbox_embed.layers[-1].bias.data, 0)
                setattr(self, 'rego_class_embed_%d_cls'%gi, nn.ModuleList([rego_class_embed_cls for _ in range(num_pred)]))
                # setattr(self, 'rego_bbox_embed_%d'%gi, nn.ModuleList([rego_bbox_embed for _ in range(num_pred)]))
                self.aux_loss = True

                for m_str in ['rcnn_net', 'layer_norms']:
                    m = getattr(self, m_str + '_%d_cls'%gi)
                    for mm in m.modules():
                        if isinstance(mm, nn.Conv2d):
                            nn.init.xavier_normal_(mm.weight)
                            nn.init.constant_(mm.bias, 0.0)
                        elif isinstance(mm, nn.Linear):
                            nn.init.xavier_normal_(mm.weight)
                            nn.init.constant_(mm.bias, 0.0)
                        elif isinstance(mm, nn.LayerNorm):
                            nn.init.constant_(mm.weight, 1.0)
                            nn.init.constant_(mm.bias, 0.0)

                m = getattr(self, 'rego_hs_fuser_%d_cls'%gi)
                for mm in m.modules():
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_normal_(mm.weight)
        ##box_branch
        self.use_rego = True
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        glimpse_transformer_box = build_glimpse_transformer()
        if self.use_rego:
            # LFOM box enlarge ratio λ; paper best is 1.75 on COCO
            self.rego_scales_box = [1.75]

            self.dropout_box = nn.Dropout(p=0.01)
            self.roi_query_dim_box = 256
            self.feat_gp_box = 4
            self.roi_feat_dim_box = self.roi_query_dim_box  # * self.feat_gp

            self.ctx_ch_box = 64
            ctx_inconvs_box = []
            ctx_outconvs_box = []
            ctx_gns_box = []
            hidden_dim = 256
            # num_classes = 20
            for i in range(4):
                for gi in range(3):
                    ctx_inconvs_box.append(
                        nn.Conv2d(256, self.ctx_ch_box, kernel_size=3, stride=1, padding=(3 + gi * 4),
                                  dilation=(3 + gi * 4), groups=8))
                    ctx_outconvs_box.append(nn.Conv2d(self.ctx_ch_box, 256, kernel_size=1, stride=1, padding=0))
                ctx_gns_box.append(nn.GroupNorm(32, 256))

            self.ctx_inconvs_box = nn.ModuleList(ctx_inconvs_box)
            self.ctx_outconvs_box = nn.ModuleList(ctx_outconvs_box)
            self.ctx_gns_box = nn.ModuleList(ctx_gns_box)
            for mm in self.ctx_inconvs_box.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.xavier_normal_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)
            for mm in self.ctx_outconvs_box.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.normal_(mm.weight, mean=0., std=1e-3)
                    nn.init.constant_(mm.bias, 0.0)

            self.roi_ext_box = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat2', 'feat3', 'feat4'], 7, 2)
            num_pred = glimpse_transformer_box.decoder.num_layers
            for gi in range(len(self.rego_scales_box)):
                rcnn_net_box = nn.Sequential(*[
                    nn.Conv2d(hidden_dim, self.roi_feat_dim_box, kernel_size=7, stride=1, padding=0,
                              groups=self.feat_gp_box),  #
                    nn.Flatten(1), nn.LayerNorm(self.roi_feat_dim_box), nn.ReLU(),
                    nn.Linear(self.roi_feat_dim_box, self.roi_query_dim_box), nn.LayerNorm(self.roi_query_dim_box)])
                setattr(self, 'rcnn_net_%d_box' % gi, rcnn_net_box)
                if gi == 0:
                    setattr(self, 'glimpse_transformer_%d_box' % gi, glimpse_transformer_box)
                else:
                    setattr(self, 'glimpse_transformer_%d_box' % gi, copy.deepcopy(glimpse_transformer_box))

                rego_hs_linear_box = nn.Linear((gi + 1) * hidden_dim, hidden_dim, bias=False)
                rego_hs_linear_norm_box = nn.LayerNorm(hidden_dim)
                setattr(self, 'rego_hs_linear_%d_box' % gi, rego_hs_linear_box)
                setattr(self, 'rego_hs_linear_norm_%d_box' % gi, rego_hs_linear_norm_box)

                rego_hs_fuser_box = nn.Linear((gi + 2) * hidden_dim, hidden_dim, bias=False)
                setattr(self, 'rego_hs_fuser_%d_box' % gi, _get_clones(rego_hs_fuser_box, num_pred))
                setattr(self, 'layer_norms_%d_box' % gi,
                        nn.ModuleList([nn.LayerNorm(hidden_dim) for i in range(num_pred)]))

                # rego_class_embed = nn.Linear(hidden_dim, num_classes)
                rego_bbox_embed_box = MLP(hidden_dim, hidden_dim, 4, 3)
                # rego_class_embed.bias.data = torch.ones(num_classes) * bias_value
                nn.init.constant_(rego_bbox_embed_box.layers[-1].weight.data, 0)
                nn.init.constant_(rego_bbox_embed_box.layers[-1].bias.data, 0)
                # setattr(self, 'rego_class_embed_%d' % gi,
                #         nn.ModuleList([rego_class_embed for _ in range(num_pred)]))
                setattr(self, 'rego_bbox_embed_%d_box' % gi,
                        nn.ModuleList([rego_bbox_embed_box for _ in range(num_pred)]))
                self.aux_loss = True

                for m_str in ['rcnn_net', 'layer_norms']:
                    m = getattr(self, m_str + '_%d_box' % gi)
                    for mm in m.modules():
                        if isinstance(mm, nn.Conv2d):
                            nn.init.xavier_normal_(mm.weight)
                            nn.init.constant_(mm.bias, 0.0)
                        elif isinstance(mm, nn.Linear):
                            nn.init.xavier_normal_(mm.weight)
                            nn.init.constant_(mm.bias, 0.0)
                        elif isinstance(mm, nn.LayerNorm):
                            nn.init.constant_(mm.weight, 1.0)
                            nn.init.constant_(mm.bias, 0.0)

                m = getattr(self, 'rego_hs_fuser_%d_box' % gi)
                for mm in m.modules():
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_normal_(mm.weight)


        self._init_layers()

    @abstractmethod
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        pass

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        outputs_class, outputs_coord, outputs_class_dn, outputs_coord_dn = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples, type = 1)
        head_inputs_dict['aux_outputs_match'] = []
        head_inputs_dict['aux_outputs_noise'] = []
        head_inputs_dict['rego_match'] = []
        head_inputs_dict['rego_noise'] = []
        ## 1:match 2:noise
        head_inputs_dict = self.rego_cls(outputs_coord, img_feats, head_inputs_dict, batch_data_samples, 1)
        head_inputs_dict = self.rego_box(outputs_coord, img_feats, head_inputs_dict, batch_data_samples, 1)

        head_inputs_dict = self.rego_cls(outputs_coord_dn, img_feats, head_inputs_dict, batch_data_samples, 2)
        head_inputs_dict = self.rego_box(outputs_coord_dn, img_feats, head_inputs_dict, batch_data_samples, 2)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples, type = 0)

        return losses
    def rego_cls(self, outputs_coord, img_feats, head_inputs_dict, batch_data_samples, type):
        if self.use_rego:
            # outputs_coord = outs[1]
            im_shapes = []
            srcs = img_feats
            for data_samples in batch_data_samples:
                shape = data_samples.batch_input_shape
                im_shapes.append(shape)
            imh, imw = im_shapes[0][-2:]
            batch_num = img_feats[-1].size(0)

            feat_dict = {}
            for i in range(len(img_feats)):
                feat_dict['feat%d' % i] = img_feats[i]

            # context features
            lvl_feats = {}
            for i in range(len(srcs)):
                feat = feat_dict['feat%d' % i]
                for gi in range(3):
                    in_feat = feat.detach()
                    local_ctx_feat = self.ctx_inconvs_cls[i * 3 + gi](F.relu(in_feat))  # N C' H W
                    local_ctx_feat = F.relu(local_ctx_feat)
                    ctx_feat = self.ctx_outconvs_cls[i * 3 + gi](local_ctx_feat)  # N C H W
                    feat = feat + 0.1 * self.dropout_cls(ctx_feat)
                feat = self.ctx_gns_cls[i](feat)
                lvl_feats.update({'feat%d' % (i + 1): feat})

            # rcnn-based glimpse
            with torch.no_grad():
                im_shape_tensor = torch.ones((batch_num, 1, 4), device=srcs[-1].device)
                for bi in range(batch_num):
                    im_shape_tensor[bi, 0, 0::2] = im_shapes[bi][1]
                    im_shape_tensor[bi, 0, 1::2] = im_shapes[bi][0]
            hs = head_inputs_dict['hidden_states_cls']
            # hs = head_inputs_dict['hidden_states_cls'] + head_inputs_dict['hidden_states_box']
            prev_dec_hs = hs[-1]
            if type == 1:
                prev_dec_hs = prev_dec_hs[:, -900:, :]
            else:
                prev_dec_hs = prev_dec_hs[:, :-900, :]
            prev_coord = outputs_coord[-1].detach()
            for gi in range(len(self.rego_scales_cls)):
                with torch.no_grad():
                    scalar_tensor = torch.ones((batch_num, 1, 4), device=srcs[-1].device)
                    scalar_tensor[:, :, 2:] = self.rego_scales_cls[gi]

                pred_bboxes = (prev_coord * scalar_tensor).clamp(max=1.0)
                pred_bboxes = pred_bboxes * im_shape_tensor
                pred_bboxes = box_ops.box_cxcywh_to_xyxy(pred_bboxes)  # N x NP x 4
                pred_bboxes = [pred_bboxes[i] for i in range(batch_num)]

                ext_roi_feat = self.roi_ext_cls(lvl_feats, pred_bboxes, [(imh, imw)])  # (Nx3NP) x C x L x L
                ext_roi_feat = getattr(self, 'rcnn_net_%d_cls' % gi)(ext_roi_feat)
                rego_in = ext_roi_feat.view(batch_num, -1, self.roi_query_dim_cls)

                prev_hs = getattr(self, 'rego_hs_linear_%d_cls' % gi)(prev_dec_hs)
                prev_hs = getattr(self, 'rego_hs_linear_norm_%d_cls' % gi)(prev_hs)

                rego_hs = getattr(self, 'glimpse_transformer_%d_cls' % gi)(rego_in, prev_hs)[0]  # NL(6) x N x Q x d
                # rego_hs = getattr(self, 'glimpse_transformer_%d'%gi)(prev_hs, rego_in)[0] # NL(6) x N x Q x d

                rego_output_classes = []
                # rego_output_coords = []
                # reference_reg = inverse_sigmoid(prev_coord)
                hs_fusers = getattr(self, 'rego_hs_fuser_%d_cls' % gi)
                l_norms = getattr(self, 'layer_norms_%d_cls' % gi)
                class_embeds = getattr(self, 'rego_class_embed_%d_cls' % gi)
                # bbox_embeds = getattr(self, 'rego_bbox_embed_%d' % gi)
                prev_h = prev_dec_hs.detach()
                for lvl in range(rego_hs.shape[0]):
                    fuse_h = torch.cat((prev_h, rego_hs[lvl]), 2)
                    fuse_h = hs_fusers[lvl](fuse_h)
                    fuse_h = l_norms[lvl](fuse_h)

                    output_class = class_embeds[lvl](fuse_h)
                    # reference_reg = reference_reg + bbox_embeds[lvl](fuse_h)
                    # output_coord = reference_reg.sigmoid()

                    rego_output_classes.append(output_class)
                    # rego_output_coords.append(output_coord)

                rego_output_classes = torch.stack(rego_output_classes)
                # rego_output_coords = torch.stack(rego_output_coords)
                rego_outs_cls = []
                if type == 1:
                    rego_out = {'pred_logits_rego_%d_match' % gi: rego_output_classes[-1]}
                    rego_outs_cls.append(rego_out)
                    head_inputs_dict['rego_match'].append(rego_outs_cls)
                else:
                    rego_out = {'pred_logits_rego_%d_dn' % gi: rego_output_classes[-1]}
                    rego_outs_cls.append(rego_out)
                    head_inputs_dict['rego_noise'].append(rego_outs_cls)
                rego_output_coords = None
                # head_inputs_dict.update(rego_outs)
                if self.training:
                    if self.aux_loss:
                        box_aux = self._set_rego_aux_loss(rego_output_classes, rego_output_coords, type,
                                                          prefix='_rego_%d' % gi)
                        if type == 1:
                            head_inputs_dict['aux_outputs_match'].append(box_aux)
                        else:
                            head_inputs_dict['aux_outputs_noise'].append(box_aux)

                # prev_dec_hs = torch.cat((prev_dec_hs, rego_hs[-1]), 2)
                # prev_coord = rego_output_coords[-1].detach()
        return head_inputs_dict

    def rego_box(self, outputs_coord, img_feats, head_inputs_dict, batch_data_samples, type):
        if self.use_rego:
            # outputs_coord = outs[1]
            im_shapes = []
            srcs = img_feats
            for data_samples in batch_data_samples:
                shape = data_samples.batch_input_shape
                im_shapes.append(shape)
            imh, imw = im_shapes[0][-2:]
            batch_num = img_feats[-1].size(0)

            feat_dict = {}
            for i in range(len(img_feats)):
                feat_dict['feat%d' % i] = img_feats[i]

            # context features
            lvl_feats = {}
            for i in range(len(srcs)):
                feat = feat_dict['feat%d' % i]
                for gi in range(3):
                    in_feat = feat.detach()
                    local_ctx_feat = self.ctx_inconvs_box[i * 3 + gi](F.relu(in_feat))  # N C' H W
                    local_ctx_feat = F.relu(local_ctx_feat)
                    ctx_feat = self.ctx_outconvs_box[i * 3 + gi](local_ctx_feat)  # N C H W
                    feat = feat + 0.1 * self.dropout_box(ctx_feat)
                feat = self.ctx_gns_box[i](feat)
                lvl_feats.update({'feat%d' % (i + 1): feat})

            # rcnn-based glimpse
            with torch.no_grad():
                im_shape_tensor = torch.ones((batch_num, 1, 4), device=srcs[-1].device)
                for bi in range(batch_num):
                    im_shape_tensor[bi, 0, 0::2] = im_shapes[bi][1]
                    im_shape_tensor[bi, 0, 1::2] = im_shapes[bi][0]
            hs = head_inputs_dict['hidden_states_box']
            # hs = head_inputs_dict['hidden_states_cls'] + head_inputs_dict['hidden_states_box']
            prev_dec_hs = hs[-1]
            if type == 1:
                prev_dec_hs = prev_dec_hs[:, -900:, :]
            else:
                prev_dec_hs = prev_dec_hs[:, :-900, :]
            prev_coord = outputs_coord[-1].detach()
            for gi in range(len(self.rego_scales_box)):
                with torch.no_grad():
                    scalar_tensor = torch.ones((batch_num, 1, 4), device=srcs[-1].device)
                    scalar_tensor[:, :, 2:] = self.rego_scales_box[gi]

                pred_bboxes = (prev_coord * scalar_tensor).clamp(max=1.0)
                pred_bboxes = pred_bboxes * im_shape_tensor
                pred_bboxes = box_ops.box_cxcywh_to_xyxy(pred_bboxes)  # N x NP x 4
                pred_bboxes = [pred_bboxes[i] for i in range(batch_num)]

                ext_roi_feat = self.roi_ext_box(lvl_feats, pred_bboxes, [(imh, imw)])  # (Nx3NP) x C x L x L
                ext_roi_feat = getattr(self, 'rcnn_net_%d_box' % gi)(ext_roi_feat)
                rego_in = ext_roi_feat.view(batch_num, -1, self.roi_query_dim_box)

                prev_hs = getattr(self, 'rego_hs_linear_%d_box' % gi)(prev_dec_hs)
                prev_hs = getattr(self, 'rego_hs_linear_norm_%d_box' % gi)(prev_hs)

                rego_hs = getattr(self, 'glimpse_transformer_%d_box' % gi)(rego_in, prev_hs)[0]  # NL(6) x N x Q x d
                # rego_hs = getattr(self, 'glimpse_transformer_%d'%gi)(prev_hs, rego_in)[0] # NL(6) x N x Q x d

                # rego_output_classes = []
                rego_output_coords = []
                reference_reg = inverse_sigmoid(prev_coord)
                hs_fusers = getattr(self, 'rego_hs_fuser_%d_box' % gi)
                l_norms = getattr(self, 'layer_norms_%d_box' % gi)
                # class_embeds = getattr(self, 'rego_class_embed_%d' % gi)
                bbox_embeds = getattr(self, 'rego_bbox_embed_%d_box' % gi)
                prev_h = prev_dec_hs.detach()
                for lvl in range(rego_hs.shape[0]):
                    fuse_h = torch.cat((prev_h, rego_hs[lvl]), 2)
                    fuse_h = hs_fusers[lvl](fuse_h)
                    fuse_h = l_norms[lvl](fuse_h)

                    # output_class = class_embeds[lvl](fuse_h)
                    reference_reg = reference_reg + bbox_embeds[lvl](fuse_h)
                    output_coord = reference_reg.sigmoid()

                    # rego_output_classes.append(output_class)
                    rego_output_coords.append(output_coord)

                # rego_output_classes = torch.stack(rego_output_classes)
                rego_output_coords = torch.stack(rego_output_coords)
                # rego_outs = []
                if type == 1:
                    rego_out = {'pred_boxes_rego_%d_match' % gi: rego_output_coords[-1]}
                    # rego_outs.append(rego_out)
                    head_inputs_dict['rego_match'][gi].append(rego_out)
                else:
                    rego_out = {'pred_boxes_rego_%d_dn' % gi: rego_output_coords[-1]}
                    # rego_outs.append(rego_out)
                    head_inputs_dict['rego_noise'][gi].append(rego_out)

                # head_inputs_dict.update(rego_outs)
                rego_output_classes = None
                if self.training:
                    if self.aux_loss:
                        box_aux = self._set_rego_aux_loss(rego_output_classes, rego_output_coords, type,
                                                          prefix='_rego_%d' % gi)
                        box_aux = box_aux[0]
                        if type == 1:
                            head_inputs_dict['aux_outputs_match'][gi].append(box_aux)
                        else:
                            head_inputs_dict['aux_outputs_noise'][gi].append(box_aux)

                # prev_dec_hs = torch.cat((prev_dec_hs, rego_hs[-1]), 2)
                # prev_coord = rego_output_coords[-1].detach()
        return head_inputs_dict


    def _set_rego_aux_loss(self, outputs_class, outputs_coord, type, prefix=''):
        if type == 1:
            if outputs_class == None:
                return [{'pred_boxes_match' + prefix: b}
                        for b in zip(outputs_coord[:-1])]
            if outputs_coord == None:
                return [{'pred_logits_match' + prefix: a}
                        for a in zip(outputs_class[:-1])]
        else:
            if outputs_class == None:
                return [{'pred_boxes_dn' + prefix: b}
                        for b in zip(outputs_coord[:-1])]
            if outputs_coord == None:
                return [{'pred_logits_dn' + prefix: a}
                        for a in zip(outputs_class[:-1])]


    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        outs = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples, type = 1)
        # outputs_class = outs[0]
        outputs_coord = outs[1]
        head_inputs_dict['rego_match'] = []
        head_inputs_dict = self.rego_cls(outputs_coord, img_feats, head_inputs_dict, batch_data_samples, 1)
        head_inputs_dict = self.rego_box(outputs_coord, img_feats, head_inputs_dict, batch_data_samples, 1)

        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)


        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.memory_key_padding_mask = torch.zeros(
                (B, memory.shape[0]),
                dtype=memory_key_padding_mask.dtype, device=memory_key_padding_mask.device
            )

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    @abstractmethod
    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Process image features before feeding them to the transformer.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              'feat_pos', and other algorithm-specific arguments.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask', and
              other algorithm-specific arguments.
        """
        pass

    @abstractmethod
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, **kwargs) -> Dict:
        """Forward with Transformer encoder.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output and other algorithm-specific
            arguments.
        """
        pass

    @abstractmethod
    def pre_decoder(self, memory: Tensor, **kwargs) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and other algorithm-specific arguments.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """
        pass

    @abstractmethod
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output, `references` including
            the initial and intermediate reference_points, and other
            algorithm-specific arguments.
        """
        pass
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
