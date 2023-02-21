import os
from os.path import join
import numpy as np
from easymocap.mytools.file_utils import write_keypoints3d, read_json, save_json
from tqdm import tqdm

skel15_to_body15 = [8, 1, 9, 12, 0, 2, 5, 10, 13, 3, 6, 11, 14, 4, 7]
panoptic15_in_body15 = [1,0,8,5,6,7,12,13,14,2,3,4,9,10,11]

if __name__ == '__main__':
    root = '/nas/home/shuaiqing/datasets/chi3d_s03'
    valdir = '/nas/home/shuaiqing/ICCV23/val/4d'
    for seq in tqdm(sorted(os.listdir(root))):
        outdir = join(root, seq, 'association_out', 'keypoints')
        gtdir = join(root, seq, 'body25')
        if not os.path.exists(outdir):
            print(seq)
            import ipdb;ipdb.set_trace()
        outnames = os.listdir(outdir)
        for outname in outnames:
            frame = int(outname.split('.')[0])
            with open(join(outdir, outname), 'r') as f:
                lines = f.readlines()
            num_person = int(lines[0])
            results = []
            for i in range(num_person):
                pid = int(lines[1+16*i])
                kpts = np.loadtxt(join(outdir, outname), skiprows=2+i*16, max_rows=15)
                kpts_body15 = np.zeros_like(kpts)
                kpts_body15[skel15_to_body15] = kpts
                results.append({
                    'id': pid,
                    'keypoints3d': kpts_body15
                })
            cvtname = join(outdir+'3d', '{:06d}.json'.format(frame))
            write_keypoints3d(cvtname, results)
            gtname = join(gtdir, '{:06d}.json'.format(frame))
            if not os.path.exists(gtname):
                continue
            if frame % 5 != 0:
                continue
            gts = read_json(gtname)
            valname = join(valdir, '{}_{:06d}.json'.format(seq, frame))
            records = {
                'gt': np.stack([np.array(gt['keypoints3d'])[panoptic15_in_body15] for gt in gts]).tolist(),
                'pred': np.stack([res['keypoints3d'][panoptic15_in_body15] for res in results]).tolist()
            }
            save_json(valname, records)