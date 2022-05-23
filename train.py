from twod_py_data import DebugXYABGenerator, XYABGenerator, VDistDataset, RoomWithRandomDisk
from models import PosToFeature, PoseToScalar
import torch
import torch.utils.data as tds
import torch.optim as optim
import time


def train_batch(model, b, optimizers, args):
    times = []
    p2fs = model["p2fs"]
    p2s = model["p2s"]
    scene_idxs, XYABs, dists = b
    
    t = time.time() 
    if args.cuda:
        scene_idxs = scene_idxs.cuda()
        XYABs = XYABs.cuda()
        dists = dists.cuda()
        torch.cuda.synchronize()
    num_points_in_batch = XYABs.shape[0]*XYABs.shape[1]
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
    for i in range(len(scene_idxs)):
        scene_idx = scene_idxs[i]
        pos = XYABs[i, :, :2]
        z = p2fs[scene_idx](pos)
        look_vec = XYABs[i, :, 2:]
        dhat = p2s(look_vec, z)
        loss += ((dhat.squeeze() - dists[i])**2).sum()
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
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--num_scenes", type=int, default=15)
    parser.add_argument("--optim_type", default="adam")
    parser.add_argument("--use_batchnorm", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    oc = {"adam": optim.Adam, "sgd":  optim.SGD}
    

    g = XYABGenerator(10)
    XYAB = g.generate(n=10)
    gd = DebugXYABGenerator(XYAB) 
    
    scenes = []
    gens = []
    for i in range(args.num_scenes):
        scenes.append(RoomWithRandomDisk(sl=10))
#        gens.append(gd)
        gens.append(XYABGenerator(10))
    V = VDistDataset(scenes, gens)

    p2f_opts = {"embedding_dim": args.hdim,
                "use_batchnorm": args.use_batchnorm,
                "num_layers": args.p2f_layers}

    p2s_opts = {"embedding_dim": args.hdim,
                "use_batchnorm": args.use_batchnorm,
                "num_layers": args.p2s_layers}

    model = {"p2fs": [PosToFeature(p2f_opts) for i in range(len(scenes))],
             "p2s": PoseToScalar(p2s_opts)}
    p2fs = model["p2fs"]
    p2s = model["p2s"]
    if args.cuda:
        for p2f in p2fs:
            p2f.cuda()
        p2s.cuda()
    
#    dataloader = tds.DataLoader(
#        dataset, collate_fn=collater, batch_size=batch_sz, shuffle=(sampler is None), sampler=sampler, drop_last=True, num_workers=args.num_workers
#    )

    dataloader = tds.DataLoader(V, batch_size=5)
    optimizers = {"p2fs": [oc[args.optim_type](p2f.parameters(), lr=args.p2f_lr) for p2f in p2fs],
                  "p2s": oc[args.optim_type](p2s.parameters(), lr=args.p2s_lr)}
    

    times = None
    for i in range(1000):
        stored_loss = 0.0
        count = 0.0
        for b in dataloader:
            count += 1
            l, t = train_batch(model, b, optimizers, args)
            if not times:
                times = t
            else:
                times = [times[i] + t[i] for i in range(len(t))]
            stored_loss += l.item()
        print(stored_loss/count)
#        print(times)
