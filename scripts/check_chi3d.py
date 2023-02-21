import os
from os.path import join

def run_cmd(cmd):
    print('[run] {}'.format(cmd))
    os.system(cmd)

if __name__ == '__main__':
    root = '/nas/home/shuaiqing/datasets/chi3d_s03'
    seqlist = []
    for seq in sorted(os.listdir(root)):
        outdir = join(root, seq, 'association_out', 'reproj')
        # if not os.path.exists(outdir):
        #     seqlist.append(seq)
        #     continue
        os.chdir('/home/qing/Code/EasyMocapPublic')
        # cmd = 'python3 apps/preprocess/extract_heatmap_by_openpose.py "{}" --pafs --restart'.format(join(root, seq))
        cmd = 'python3 scripts/dataset/parse_4dassociation.py "{}"'.format(join(root, seq))
        run_cmd(cmd)
        os.chdir('/home/qing/Code/4d_association/build')
        cmd = './bin/mocap "{}" chi3d'.format(join(root, seq))
        run_cmd(cmd)
        continue
        outnames = os.listdir(outdir)
        imgdir = join(root, seq, 'images', '50591643')
        imgnames = os.listdir(imgdir)
        if len(outnames) != len(imgnames):
            print(seq, len(outnames), len(imgnames))
            seqlist.append(seq)
        