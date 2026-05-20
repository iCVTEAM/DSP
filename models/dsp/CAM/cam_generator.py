from .pytorch_grad_cam import GradCAM
from .pytorch_grad_cam.utils.image import scale_cam_image
from . import clip
from .utils import reshape_transform, zeroshot_classifier, ClipOutputTarget, scoremap2bbox

from torchvision import transforms
import torch
import numpy as np
import cv2

class CAMGenerator:
    def __init__(self, categories, clip_path):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_path = clip_path
        self.clip_model, _ = clip.load(self.clip_path, device="cpu")
        self.target_layers = [self.clip_model.visual.transformer.resblocks[-1].ln_1]
        self.cam = GradCAM(model=self.clip_model, target_layers=self.target_layers, reshape_transform=reshape_transform, use_cuda=True)

        self.categories = categories
        # if categories is None:
        #     self.categories = ['vehicle', 'baseballfield', 'groundtrackfield', 'windmill', 'bridge', 'overpass', 'ship', 'airplane', 'tenniscourt', 'airport',
        #                     'expressway-service-area', 'basketballcourt', 'stadium', 'storagetank', 'chimney', 'dam', 'expressway-toll-station', 'golffield', 'trainstation', 'harbor']
        self.background_categories = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet', 
                                      'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',]

        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def _prepare(self):
        self.bg_text_features = zeroshot_classifier(self.background_categories, ['a clean origami {}.'], self.clip_model, self.device)
        self.fg_text_features = zeroshot_classifier(self.categories, ['a clean origami {}.'], self.clip_model, self.device)

    def to(self, device, dtype):
        self.device = device
        self.clip_model.to(device)
        self.cam.set_device(device)
        self._prepare()

    def re_normalize(self, image):
        # image = (image / 2) + 0.5
        image = self.normalize(image)
        return image
    
    def get_label_list(self, captions):
        label_list = []
        label_id_list = []
        for caption in captions:
            if caption in self.categories and caption not in label_list:
                label_list.append(caption)
                label_id_list.append(self.categories.index(caption))
        return label_list, label_id_list
    
    def get_label_list_with_bboxes(self, captions, bboxes):
        label_list, label_id_list, bboxes_list = [], [], []
        for caption, bbox in zip(captions, bboxes):
            if caption not in self.categories:
                continue
            if caption not in label_list:
                label_list.append(caption)
                label_id_list.append(self.categories.index(caption))
                bboxes_list.append([bbox])
            else:
                bboxes_list[label_list.index(caption)].append(bbox)
        return label_list, label_id_list, bboxes_list

    def __call__(self, image, captions, bboxes, gt_bboxes_only=False):
        image = self.re_normalize(image)
        # label_list, label_id_list = self.get_label_list(captions[0][1:])
        label_list, label_id_list, bboxes_list = self.get_label_list_with_bboxes(captions[0][1:], bboxes[0])
        h, w = image.shape[-2], image.shape[-1]
        image_features, attn_weight_list = self.clip_model.encode_image(image, h, w)

        bg_features_temp = self.bg_text_features
        fg_features_temp = self.fg_text_features[label_id_list]
        text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
        input_tensor = [image_features, text_features_temp, h, w]

        keys, refined_cam_list = [], []
        for idx, (label, bbox) in enumerate(zip(label_list, bboxes_list)):
            keys.append(self.categories.index(label))
            targets = [ClipOutputTarget(label_list.index(label))]
            grayscale_cam, logits_per_image, attn_weight_last = self.cam(input_tensor=input_tensor, targets=targets, target_size=None)
            grayscale_cam = grayscale_cam[0, :] # [32, 32]
            # grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))

            if idx == 0:
                attn_weight_list.append(attn_weight_last)
                attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
                attn_weight = torch.stack(attn_weight, dim=0)[-8:] # [8, 1, 1024, 1024]
                attn_weight = torch.mean(attn_weight, dim=0) # [1, 1024, 1024]
                # attn_weight = attn_weight[0].detach() # [1024, 1024] # original detach
                attn_weight = attn_weight[0] #.detach() # [1024, 1024]
            attn_weight = attn_weight.float()

            gt_box, gt_cnt = (np.array(bbox) * [grayscale_cam.shape[1], grayscale_cam.shape[0], grayscale_cam.shape[1], grayscale_cam.shape[0]]).astype(int), len(bbox)
            if gt_bboxes_only:
                box, cnt = gt_box, gt_cnt
            else:
                box, cnt = scoremap2bbox(scoremap=grayscale_cam.cpu().data.numpy(), threshold=0.4, multi_contour_eval=True)
                box, cnt = np.concatenate([box, gt_box], axis=0), cnt + gt_cnt
            aff_mask = torch.zeros_like(grayscale_cam)
            for i_ in range(cnt):
                x0_, y0_, x1_, y1_ = box[i_]
                aff_mask[y0_:y1_, x0_:x1_] = 1
            aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])

            aff_mat = attn_weight
            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
            for _ in range(2):
                trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
            trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
            for _ in range(1):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            trans_mat = trans_mat * aff_mask

            cam_to_refine = grayscale_cam.view(-1, 1)
            cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
            cam_refined = cam_refined - cam_refined.min()
            cam_refined = cam_refined / (cam_refined.max() + 1e-7)
            refined_cam_list.append(cam_refined)

        keys = torch.tensor(keys)
        refined_cams = torch.stack(refined_cam_list, dim=0)

        return refined_cams, keys


if __name__ == '__main__':
    cam = CAMGenerator()
    