from twod_py_data import DebugXYABGenerator, XYABGenerator, TagDataset, RandomDisksAndTriangles
from models import PosToFeature, PoseToScalar
import torch
import torch.nn as nn
import torch.utils.data as tds
import torch.optim as optim
import time


def train_batch(model, b, optimizers, args):
    times = []
    p2fs = model["p2fs"]
    p2s = model["p2s"]
    scene_idxs, XYs, tags = b
    
    t = time.time() 
    if args.cuda:
        scene_idxs = scene_idxs.cuda()
        XYs = XYs.cuda()
        tags = tags.cuda()
        torch.cuda.synchronize()
    num_points_in_batch = XYs.shape[0]*XYs.shape[1]
    times.append(time.time()- t)

    t = time.time() 
    for o in optimizers["p2fs"]:
        o.zero_grad()
    optimizers["p2s"].zero_grad()
    for p2f in p2fs:
        p2f.train()
    if args.cuda:
        torch.cuda.synchronize()
    times.append(time.time()- t)

    t = time.time()
    loss = 0
    bce = nn.BCELoss() 
    for i in range(len(scene_idxs)):
        scene_idx = scene_idxs[i]
        pos = XYs[i, :, :2]
        z = p2fs[scene_idx](pos)
        dhat = p2s(z)
        loss += bce(dhat.squeeze(),tags[i])
    loss /= num_points_in_batch
    if args.cuda:
        torch.cuda.synchronize()
    times.append(time.time()- t)

    t = time.time()
    loss.backward()
    if args.cuda:
        torch.cuda.synchronize()
    times.append(time.time()- t)

    t = time.time()
    for o in optimizers["p2fs"]:
        o.step()
    optimizers["p2s"].step()
    if args.cuda:
        torch.cuda.synchronize()
    times.append(time.time()- t)
    
    return loss, times
             




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2f_lr", type=float, default=.0001)
    parser.add_argument("--p2s_lr", type=float, default=.0001)
    parser.add_argument("--p2f_layers", type=int, default=4)
    parser.add_argument("--p2s_layers", type=int, default=4)
    parser.add_argument("--fourier_embedding_dim", type=int, default=-1)
    parser.add_argument("--fourier_embedding_scale", type=float, default=2.0)
    parser.add_argument("--locs_per_scene", type=int, default=1024)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--num_scenes", type=int, default=15)
    parser.add_argument("--optim_type", default="adam")
    parser.add_argument("--use_batchnorm", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    oc = {"adam": optim.Adam, "sgd":  optim.SGD}

    tags = ["obstacle", "disk", "triangle"]

    scenes = []
    gens = []
    for i in range(args.num_scenes):
        scenes.append(RandomDisksAndTriangles({"sl":10}))
        gens.append(XYABGenerator(10))
    T = TagDataset(scenes, tags, gens, samples_per_scene=args.locs_per_scene)

    p2f_opts = {"embedding_dim": args.hdim,
                "use_batchnorm": args.use_batchnorm,
                "num_layers": args.p2f_layers,
                "fourier_embedding_dim": args.fourier_embedding_dim,
                "fourier_embedding_scale": args.fourier_embedding_scale
    }

#    p2s_opts = {"embedding_dim": args.hdim,
#                "use_batchnorm": args.use_batchnorm,
#                "num_layers": args.p2s_layers,
#                "fourier_embedding_dim": args.fourier_embedding_dim,
#                "fourier_embedding_scale": args.fourier_embedding_scale,
#                "output_nonlin": "sigmoid"
#    }

    model = {"p2fs": [PosToFeature(p2f_opts) for i in range(len(scenes))],
             "p2s":nn.Sequential(nn.Linear(args.hdim, len(tags)), nn.Sigmoid())
    #         "p2s": PoseToScalar(p2s_opts)
    }
    p2fs = model["p2fs"]
    p2s = model["p2s"]
    if args.cuda:
        for p2f in p2fs:
            p2f.cuda()
            #fixme why is this not propagating?
            try:
                p2f.embedding[0].cuda()
            except:
                pass
        p2s.cuda()
        #fixme why is this not propagating?
        try:
            p2s.embedding[0].cuda()
        except:
            pass
            
    optimizers = {"p2fs": [oc[args.optim_type](p2f.parameters(), lr=args.p2f_lr) for p2f in p2fs],
                  "p2s": oc[args.optim_type](p2s.parameters(), lr=args.p2s_lr)}
    

    times = None
    for i in range(1000):
        stored_loss = 0.0
        count = 0.0
        for j in range(200):
#        for b in dataloader:
            b = T[j]
            count += 1
            l, t = train_batch(model, b, optimizers, args)
            if not times:
                times = t
            else:
                times = [times[i] + t[i] for i in range(len(t))]
            stored_loss += l.item()
        print(stored_loss/count)
        print(times)
