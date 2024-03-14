import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class PatternSynthesizer(Synthesizer):
    # pattern_tensor: torch.Tensor = torch.tensor([
    #     [1., 0., 1.],
    #     [-10., 1., -10.],
    #     [-10., -10., 0.],
    #     [-10., 1., -10.],
    #     [1., 0., 1.]
    # ])

    pattern_tensors = {0: torch.Tensor([[-10.,  1., -10.],
                        [  1.,  0.,   1.],
                        [  1.,  0.,   1.],
                        [  1.,  0.,   1.],
                        [-10.,  1., -10.]]), 1: torch.Tensor([[-10.,  1., -10.],
                        [-10.,  1., -10.],
                        [-10.,  1., -10.],
                        [-10.,  1., -10.],
                        [-10.,  1., -10.]]), 2: torch.Tensor([[  1.,  1.,   1.],
                        [-10., -10.,  1.],
                        [  1.,  1.,   1.],
                        [  1., -10., -10.],
                        [  1.,  1.,   1.]]), 
                        3: torch.Tensor([[  1.,  1.,   1.],
                        [-10., -10.,  1.],
                        [-10.,  1.,   1.],
                        [-10., -10.,  1.],
                        [  1.,  1.,   1.]]), 
                        4: torch.Tensor([[  1., -10.,  1.],
                        [  1., -10.,  1.],
                        [  1.,  1.,   1.],
                        [-10., -10.,  1.],
                        [-10., -10.,  1.]]), 
                        5: torch.Tensor([[  1.,  1.,   1.],
                        [  1., -10., -10.],
                        [  1.,  1.,   1.],
                        [-10., -10.,  1.],
                        [  1.,  1.,   1.]]),
                        6: torch.Tensor([[-10.,  1.,  1.],
                        [  1., -10., -10.],
                        [  1.,  1.,   1.],
                        [  1., -10.,  1.],
                        [-10.,  1.,  1.]]),
                        7: torch.Tensor([[  1.,  1.,   1.],
                        [-10., -10.,  1.],
                        [-10., -10.,  1.],
                        [-10., -10.,  1.],
                        [-10., -10.,  1.]]), 
                        8: torch.Tensor([[-10.,  1., -10.],
                        [  1., -10.,  1.],
                        [-10.,  1., -10.],
                        [  1., -10.,  1.],
                        [-10.,  1., -10.]]),
                        9: torch.Tensor([[-10.,  1.,  1.],
                        [  1., -10.,  1.],
                        [-10.,  1.,  1.],
                        [-10., -10.,  1.],
                        [  1.,  1., -10.]])}
    
    "Just some random 2D pattern."

    x_top = 0
    "X coordinate to put the backdoor into."
    y_top = 0
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: Task):
        super().__init__(task)
        # self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)

    def make_pattern(self, pattern_tensor, x_top, y_top):
        full_image = torch.zeros(self.params.input_shape)
        full_image.fill_(self.mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        if x_bot >= self.params.input_shape[1] or \
                y_bot >= self.params.input_shape[2]:
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {self.params.input_shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
        
        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)
        # self.pattern = self.task.normalize(full_image).to(self.params.device)

    def synthesize_inputs(self, batch, attack_portion=None, test_phase=False):
        import numpy as np
        blend_ratio = 0.02

        def sample_negative_labels(label, n_classes):
            label_cpu = label.detach().cpu().numpy()
            neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
            neg_label = torch.tensor(np.array(neg_label))
            return neg_label.to("cuda")
        atktarget = sample_negative_labels(batch.labels[:attack_portion], n_classes=10)
        batch.labels[:attack_portion] = atktarget
        

        # pattern, mask = self.get_pattern()
        # pattern, mask = self.get_pattern()
        # batch.inputs[:attack_portion] = (1 - mask) * \
        #                                 batch.inputs[:attack_portion] + \
        #                                 mask * pattern

        for i, label in enumerate(batch.labels):
            full_image = torch.zeros(self.params.input_shape)
            full_image.fill_(self.mask_value)

            pattern_tensor = self.pattern_tensors[label.item()]
            x_bot = self.x_top + pattern_tensor.shape[0]
            y_bot = self.y_top + pattern_tensor.shape[1]

            full_image[:, self.x_top:x_bot, self.y_top:y_bot] = pattern_tensor
            mask = 1 * (full_image != self.mask_value).to(self.params.device)
            # pattern = self.task.normalize(full_image).to(self.params.device)
            pattern = full_image.to(self.params.device)
            # pattern = torch.clamp(pattern, 0.0, 1.0)

            if not test_phase:
                # batch.inputs[i] = (1 - blend_ratio)*(1 - mask) * batch.inputs[i] + blend_ratio * mask * pattern
                # batch.inputs[i] = (1 - blend_ratio) * batch.inputs[i] + blend_ratio * pattern
                # batch.inputs[i] = torch.clamp(batch.inputs[i], 0.0, 1.0)
                batch.inputs[i] = (1 - blend_ratio) * batch.inputs[i] + blend_ratio * mask * pattern
                # batch.inputs[i] = batch.inputs[i] + mask * pattern
                batch.inputs[i] = torch.clamp(batch.inputs[i], 0.0, 1.0)
                # batch.inputs[i] = (1 - mask) * batch.inputs[i] + mask * pattern
                # batch.inputs[i] = batch.inputs[i]
            else:
                # batch.inputs[i] = (1 - mask) * batch.inputs[i] + mask * pattern
                batch.inputs[i] = batch.inputs[i] + mask * pattern
                batch.inputs[i] = torch.clamp(batch.inputs[i], 0.0, 1.0)
                # batch.inputs[i] = mask * pattern

        return

    def synthesize_labels(self, batch, attack_portion=None, test_phase=None):
        # batch.labels[:attack_portion].fill_(self.params.backdoor_label)
        # batch.labels.fill_(self.params.backdoor_label)
        return

    def get_pattern(self):
        # if self.params.backdoor_dynamic_position:
        #     resize = random.randint(self.resize_scale[0], self.resize_scale[1])
        #     pattern = self.pattern_tensor
        #     if random.random() > 0.5:
        #         pattern = functional.hflip(pattern)
        #     image = transform_to_image(pattern)
        #     pattern = transform_to_tensor(
        #         functional.resize(image,
        #             resize, interpolation=0)).squeeze()

        #     x = random.randint(0, self.params.input_shape[1] \
        #                        - pattern.shape[0] - 1)
        #     y = random.randint(0, self.params.input_shape[2] \
        #                        - pattern.shape[1] - 1)
        #     self.make_pattern(pattern, x, y)

        return self.pattern, self.mask
