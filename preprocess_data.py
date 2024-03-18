import itk
import torch
import torch.nn.functional as F
import numpy as np
import footsteps
footsteps.initialize()

data = []
for name in glob.glob("data/auto_files_resampled/*.tif")[:2]:
    img = itk.imread(name)
    img = np.array(img)
    img = torch.tensor(img).float()
    img = F.avg_pool3d(img, 4)
    img = img[:137, :280, :106]
    assert(img.shape == (137, 280, 106))
    img = img[None, None]
    data.append(img)


torch.save(data, footsteps.output_dir + "downsampled_imgs.trch")
