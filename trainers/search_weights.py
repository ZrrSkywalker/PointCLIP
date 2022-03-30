import os.path as osp
import torch

import clip


modelnet40_classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

CUSTOM_TEMPLATES_ZS = {
    'ModelNet40': 'point cloud depth map of a {}.',
}

CUSTOM_TEMPLATES_FS = {
    'ModelNet40': 'point cloud of a big {}.',
}

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def textual_encoder(cfg, classnames, templates, clip_model):
    
    temp = templates[cfg.DATASET.NAME]
    prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
    prompts = torch.cat([clip.tokenize(p) for p in prompts])
    prompts = prompts.cuda()
    text_feat = clip_model.encode_text(prompts).repeat(1, cfg.MODEL.PROJECT.NUM_VIEWS)
    return text_feat


@torch.no_grad()
def search_weights_zs(cfg):

    print("\n***** Searching for view weights *****")

    image_feat = torch.load(osp.join(cfg.OUTPUT_DIR, "features.pt"))
    labels = torch.load(osp.join(cfg.OUTPUT_DIR, "labels.pt"))

    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME)
    clip_model.eval()

    text_feat = textual_encoder(cfg, modelnet40_classes, CUSTOM_TEMPLATES_ZS, clip_model)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  

  
    # Before search
    logits = clip_model.logit_scale.exp() * image_feat @ text_feat.t() * 1.0
    acc, _ = accuracy(logits, labels, topk=(1, 5))
    
    acc = (acc / image_feat.shape[0]) * 100
    print(f"=> Before search, PointCLIP accuracy: {acc:.2f}")


    # Search
    print("Start to search:")

    best_acc = 0
    # Search_time can be modulated in the config for faster search
    search_time, search_range = cfg.SEARCH.TIME, cfg.SEARCH.RANGE
    search_list = [(i + 1) * search_range / search_time  for i in range(search_time)]

    for a in search_list:
        for b in search_list:
            for c in search_list:
                for d in search_list:
                    for e in search_list:
                        for f in search_list:
                            # Reweight different views
                            view_weights = torch.tensor([a, b, c, d, e, f]).cuda()
                            image_feat_w = image_feat.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS, cfg.MODEL.BACKBONE.CHANNEL) * view_weights.reshape(1, -1, 1)
                            image_feat_w = image_feat_w.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS * cfg.MODEL.BACKBONE.CHANNEL).type(clip_model.dtype)
                            
                            logits = clip_model.logit_scale.exp() * image_feat_w @ text_feat.t() * 1.0
                            acc, _ = accuracy(logits, labels, topk=(1, 5))
                            acc = (acc / image_feat.shape[0]) * 100

                            if acc > best_acc:
                                print('New best accuracy: {:.2f}, view weights: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(acc, a, b, c, d, e, f))
                                best_acc = acc

    print(f"=> After search, PointCLIP accuracy: {best_acc:.2f}")


@torch.no_grad()
def search_weights_fs(cfg):

    print("\n***** Searching for view weights *****")

    image_feat = torch.load(osp.join(cfg.OUTPUT_DIR, "features.pt"))
    labels = torch.load(osp.join(cfg.OUTPUT_DIR, "labels.pt"))

    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME)
    clip_model.eval()

    text_feat = textual_encoder(cfg, modelnet40_classes, CUSTOM_TEMPLATES_FS, clip_model)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  

  
    # Before search
    logits = clip_model.logit_scale.exp() * image_feat @ text_feat.t() * 1.0
    acc, _ = accuracy(logits, labels, topk=(1, 5))
    
    acc = (acc / image_feat.shape[0]) * 100
    print(f"=> Before search, PointCLIP accuracy: {acc:.2f}")


    # Search
    print("Start to search:")

    best_acc = 0
    # Search_time can be modulated in the config for faster search
    search_time, search_range = cfg.SEARCH.TIME, cfg.SEARCH.RANGE
    search_list = [(i + 1) * search_range / search_time  for i in range(search_time)]

    for a in search_list:
        for b in search_list:
            for c in search_list:
                for d in search_list:
                    for e in search_list:
                        for f in search_list:
                            for g in search_list:
                                for h in search_list:
                                    for i in search_list:
                                        for j in search_list:
                                            # Reweight different views
                                            view_weights = torch.tensor([a, b, c, d, e, f, g, h, i, j]).cuda()
                                            image_feat_w = image_feat.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS, cfg.MODEL.BACKBONE.CHANNEL) * view_weights.reshape(1, -1, 1)
                                            image_feat_w = image_feat_w.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS * cfg.MODEL.BACKBONE.CHANNEL).type(clip_model.dtype)
                                            
                                            logits = clip_model.logit_scale.exp() * image_feat_w @ text_feat.t() * 1.0
                                            acc, _ = accuracy(logits, labels, topk=(1, 5))
                                            acc = (acc / image_feat.shape[0]) * 100

                                            if acc > best_acc:
                                                print('New best accuracy: {:.2f}, view weights: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(acc, a, b, c, d, e, f, g, h, i, j))
                                                best_acc = acc

    print(f"=> After search, PointCLIP accuracy: {best_acc:.2f}")

