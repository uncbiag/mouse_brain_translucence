import itk
import os
from datetime import datetime

import footsteps
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration import config
from icon_registration.losses import ICONLoss, to_floats
from icon_registration.mermaidlite import compute_warped_image_multiNC
import icon_registration.itk_wrapper



from train_knee import make_net

def get_model():
    input_shape = [1, 1, 105, 280, 135]
    net = make_net(input_shape)
    from os.path import exists
    weights_location = "network_weights/network_weights_86100"
    if not exists(weights_location):
        print("Downloading pretrained model")
        import urllib.request
        import os
        download_path = "https://github.com/uncbiag/mouse_brain_translucence/releases/download/constricon_weights/network_weights_86100"
        os.makedirs("network_weights/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"))
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net

def preprocess(image):
    image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    
    return image

if __name__ == "__main__":
    import itk
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed")
    parser.add_argument("--moving")
    parser.add_argument("--transform_out")
    parser.add_argument("--warped_moving_out", default=None)

    args = parser.parse_args()

    net = get_model()

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)

    phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(net,preprocess(moving), preprocess(fixed), finetune_steps=50)

    itk.transformwrite([phi_AB], args.transform_out)

    if args.warped_moving_out:
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
        warped_image_A = itk.resample_image_filter(
                moving,
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed
                )
        itk.imwrite(warped_image_A, args.warped_moving_out)





