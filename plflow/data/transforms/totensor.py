import torch


class RGBToTensor:
    """
    removed branching in to_tensor operations
    """

    def __init__(self):
        pass

    def __call__(self, pic, *args, **kwargs):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        img = img.permute((2, 0, 1)).contiguous()
        return img.to(dtype=torch.float32).div(255)
