import itk
import glob
import torch
import torch.nn.functional as F
import numpy as np
import footsteps
footsteps.initialize()

data = []
for name in glob.glob("data/auto_files_resampled/*.tif"):
    img = itk.imread(name)
    img = np.array(img).astype(int)
    img = torch.tensor(img).float()[None, None].cuda()
    img = F.avg_pool3d(img,2)
    img = F.interpolate(img, [105, 280, 135])
    assert(img.shape == (1, 1, 105, 280, 135))
    data.append(img.cpu())


torch.save(data, footsteps.output_dir + "downsampled_imgs.trch")
