import torch
import torch.utils.data as tds
import numpy as np
import matplotlib.pyplot as plt
import argparse

INF = 100
pi = np.pi

class VDist:
    """
    class to get signed distances from a point (x, y)
    in a direction (a,b) to the nearest point in a "scene".
    a scene consists of regions of space that should be considered 
    obstacles.  the VDist object implements a method
    
    .get_vdist(x, y, a, b) that returns the distance to the 
        nearest point in the obstacle set to x,y along the oriented ray a,b.
        if there is no obstacle along this ray, it returns a fixed maximum self.inf

    .get_vdist_batch(XYAB) should have the same output as get_vdist(x,y, a,b)
        for each row of the Nx4 torchTensor XYAB; where the four entries of the row are
        taken to be x, y, a, b
        outputs an N Tensor
    """
    def get_vdist(self, x, y, a, b):
        pass
    
    def set_inf(self, v):
        self.inf = v

    def get_vdist_batch(self, XYAB):
        """ 
        input an N x 4 array of XYAB
        overload me for parallel
        """
        N = XYAB.shape[0]
        out = torch.zeros(N)
        for i in range(XYAB.shape[0]):
            out[i] = self.get_vdist(*XYAB[i])
        return out
        
    def raycast_image(self, imsize, sl, num_rays_per_point):        
        a = np.zeros((imsize,imsize,3))
        l = torch.linspace(-sl, sl, imsize)
        idxs = torch.LongTensor(list(range(imsize)))
        angles = [2*pi*t/num_rays_per_point for t in range(num_rays_per_point)]
        gx, gy, ga = torch.meshgrid(l, l, torch.Tensor(angles))
        gix, giy, _ = torch.meshgrid(idxs, idxs, torch.LongTensor(angles))
        XYAB = torch.stack([gx.reshape(-1),
                            gy.reshape(-1),
                            torch.cos(ga.reshape(-1)),
                            torch.sin(ga.reshape(-1))],
                           1)
        d = self.get_vdist_batch(XYAB)
        gix = gix.reshape(imsize**2, num_rays_per_point)
        giy = giy.reshape(imsize**2, num_rays_per_point)
        d = d.reshape(imsize**2, num_rays_per_point)
        d, _ = d.min(1)
        d = d - d.min()
        d = d/d.max()
        for i in range(imsize**2):
            a[giy[i, 0], gix[i, 0], 0] = 1 - d[i]
            a[giy[i, 0], gix[i, 0], 2] = d[i]
        return a

class UnionVDist(VDist):
    def __init__(self, members=None, inf=INF):
        self.members = [m for m in members]
        self.set_inf(inf)
        
    def append(vdist):
        self.members.append(vdist)
        
    def get_vdist(self, x, y, a, b):
        dists = [m.get_vdist(x,y,a,b) for m in self.members]
        dists = [v if v > 0 else self.inf for v in dists]
        dists = [v if v < self.inf else self.inf for v in dists] 
        return min(dists)

    def get_vdist_batch(self, XYAB):
        dists = torch.stack([m.get_vdist_batch(XYAB) for m in self.members])
        dists[dists < 0] = self.inf
        dists[dists > self.inf] = self.inf
        md, _ = dists.min(axis=0)
        return md
    
class OrientedLine(VDist):
    def __init__(self, v, w, r, inf=INF):
        # vx + wy >r means oob        
        self.v = v
        self.w = w
        self.r = r
        self.set_inf(inf)

    def get_vdist(self, x, y, a, b):
        ip = a*self.v + b*self.w
        if ip == 0:
            return self.inf
        t = (self.r - x*self.v - y*self.w)/ip
        if t < 0:
            return self.inf
        return np.sqrt(a**2 + b**2) * t
    
    def get_vdist_batch(self, XYAB):
        vw = torch.Tensor([self.v, self.w])
        ip = XYAB[:,2:]@vw
        orth = ip == 0
        ip[orth] = 1.0
        t = (self.r - XYAB[:,:2]@vw)/ip
        t[orth] = self.inf
        t[t<0] = self.inf
        return XYAB[:,2:].norm(2,1) * t


    

class BasicRoom(UnionVDist):
    def __init__(self, sl=10):
        normals = [(1, 0), (-1, 0), (0, 1), (0, -1)] 
        super().__init__([OrientedLine(n[0], n[1], sl) for n in normals])


class RoomWithRandomDisk(UnionVDist):
    def __init__(self, sl=10):
        normals = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        walls = [OrientedLine(n[0], n[1], sl) for n in normals]
        r = np.random.uniform(0, sl/2)
        x =  np.random.uniform(-sl, sl)
        y =  np.random.uniform(-sl, sl)
        D = Disk(x, y, r)
        super().__init__(walls + [D])


class Disk(VDist):
    def __init__(self, cx, cy, r, inf=INF):
        
        self.set_inf(inf)
        self.cx = cx
        self.cy = cy
        self.r = r
        self.rs = r**2

    def get_intersect(self, x, y, a, b):
        dx = x - self.cx
        dy = y - self.cy
        A = a**2 + b**2
        B = 2*(a*dx + b*dy)
        C = dx**2 + dy**2 - self.rs
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            return None
        else:
            sd = np.sqrt(discriminant)
            if B > 0:
                t = -B + sd
            else:
                if -B < sd:
                    t = -B + sd
                else:
                    t = -B - sd
            if t < 0:
                return None
            t = t/(2*A)
            return x + t*a, y + t*b 
        
    def get_vdist(self, x, y, a, b):
        z = self.get_intersect(x, y, a, b)
        if z is None:
            return self.inf
        else:
            return np.sqrt((z[0] - x)**2 + (z[1] - y)**2)

    def get_intersect_batch(self, XYAB):
        c = torch.Tensor([self.cx, self.cy])
        n = XYAB.shape[0]
        dxdy = XYAB[:, :2] - c.expand(n, 2)
        a = (XYAB[:,2:]**2).sum(1)
        b = (XYAB[:,2:]*dxdy).sum(1)*2
        c = (dxdy**2).sum(1) - self.rs
        dmt = b**2 - 4*a*c
        notok_mask = dmt < 0
        dmt[notok_mask] = 0
        sqtdmt = torch.sqrt(dmt) 
        neg_b = b <= 0
        t = -b
        sqtdmt[neg_b*(-b >= sqtdmt)] = -sqtdmt[neg_b*(-b >= sqtdmt)] 
        t = t + sqtdmt
        notok_mask[t < 0] = True
        notok_mask[a==0] = True
        t = t/(2*a)
        return XYAB[:,:2] + XYAB[:, 2:]*t.unsqueeze(1).expand(n,2), notok_mask

    def get_vdist_batch(self, XYAB):
        z, notok_mask = self.get_intersect_batch(XYAB)
        d = (z - XYAB[:,:2]).norm(2,1)
        d[notok_mask] = self.inf
        return d



class XYABGenerator:
    """
    basic class for generating random x, y, a, b tuples.
    x, y will be drawn from uniform -sl, sl
    a ,b uniform over unit circle
    """    
    def __init__(self, sl):
        self.sl = sl

    def generate(self, n=1):
        xy = 2*self.sl*torch.rand((n,2)) - self.sl
        ab = torch.randn((n,2))
        ab = ab/ab.norm(2,1).unsqueeze(1)
        return torch.cat([xy, ab],1)

class DebugXYABGenerator:
    """
    always returns a fixed set of xyab 
    """
    def __init__(self, XYAB):
        self.XYAB = XYAB

    def generate(self, n=1):
        #ignores n
        return self.XYAB
    
    
class VDistDataset(tds.Dataset):
    def __init__(self, vdists, xyab_generators, N=1000, samples_per_scene=10):
        self.vdists = vdists
        # TODO make sure generators and scenes match up w.r.t. sl etc
        self.xyzab_generators = xyab_generators

        # size of an epoch:
        self.N = N
        
        self.num_scenes = len(self.vdists)
        self.samples_per_scene = samples_per_scene
        
    def __getitem__(self, index):
        # ignoring index...
        vidx = torch.randint(0, self.num_scenes, (1,))
        S = self.vdists[vidx]
        G = self.xyzab_generators[vidx]
        XYAB = G.generate(n=self.samples_per_scene)
        return vidx, XYAB, S.get_vdist_batch(XYAB)

    def __len__(self):
        return self.N

    
if __name__ == "__main__":
    def get_and_show_im(V):
        im = V.raycast_image(100, 10, 100)
        plt.imshow(im)
        plt.show()

    A = BasicRoom()
    D0 = Disk(0, 1, 1)
    R = RoomWithRandomDisk(sl=10)

    scenes = []
    gens = []
    for i in range(15):
        scenes.append(RoomWithRandomDisk(sl=10))
        gens.append(XYABGenerator(10))
    V = VDistDataset(scenes, gens)
    

#    x[:,:40, 2]=1
#plt.show(x)
#p x
#x
#plt.show(x)
#plt.imshow(x)
#plt.show()
