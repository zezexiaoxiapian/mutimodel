import glob
from os import path
from pycocotools.coco import COCO
import random
def img_list(dataset):
    rbg_imgs = glob.glob('%s/RGB/*.jpg' % dataset)
    the_imgs = set(glob.glob('%s/thermal_8_bit/*.jpeg' % dataset))
    filterd_rbg_imgs = []
    for p in rbg_imgs:
        if p.replace('RGB', 'thermal_8_bit').replace('.jpg', '.jpeg') in the_imgs:
            filterd_rbg_imgs.append(path.abspath(p))
    return filterd_rbg_imgs
coco = COCO('/home/tjz/FLIR-tiny/val/thermal_annotations.json')
ind = 0
imInfo = coco.imgs[ind]
annIds = coco.getAnnIds(imgIds=imInfo['id'])
anns = coco.loadAnns(annIds)
i=0
for i in range(len(anns[0])):
    print(anns[0]["category_id"])
    i = i+1
sets = ['train', 'val']
for s in sets:
    il = img_list(s)
    with open('%s.txt' % s, 'w') as fw:
        fw.write('\n'.join(il))