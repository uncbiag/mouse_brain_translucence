import footsteps
import random
import icon_registration as icon
import torch

import train_knee

if __name__ == "__main__":
    input_shape = [1, 1, 105, 280, 135]
    footsteps.initialize()

    dataset = torch.load(
        "results/preprocessed_data/downsampled_imgs.trch"
    )

    BATCH_SIZE=3
    GPUS=4

    def quantile(arr: torch.Tensor, q):
        arr = arr.flatten()

        l = len(arr)

        return torch.kthvalue(arr, int(q * l)).values
        


    def halfbatch():
        img = torch.cat([random.choice(dataset) for i in range(GPUS * BATCH_SIZE)])
        img = img.cuda()
        img = img - quantile(img, .01)
        img = img / quantile(img, .99)
        return img

    loss = train_knee.make_net(input_shape=input_shape)

    net_par = torch.nn.DataParallel(loss).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()
    icon.train_batchfunction(net_par, optimizer, lambda: (halfbatch(), halfbatch()), unwrapped_net=loss)
