import os
import numpy as np
import torch
from torchvision import transforms
from datasets.random_erasing import RandomErasing
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import datasets.video_transforms as video_transforms 
import datasets.volume_transforms as volume_transforms
import pandas as pd
import random



class SSVideoClsDataset(Dataset):
    def __init__(self, 
                filenames,
                labels,
                mode='train',
                keep_aspect_ratio=True, height=256, width=320, short_side_size=256, crop_size=224,
                num_frames=16, 
                eval_num_crops=3, eval_num_segments=2,
                aug_args=None):

        # train, val or eval
        self.mode = mode
        # size for original video and model input video img size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.height = height
        self.width = width
        self.short_side_size = short_side_size
        self.crop_size = crop_size
        # frame selection
        self.num_frames = num_frames
        self.eval_num_crops = eval_num_crops
        self.eval_num_segments = eval_num_segments
        # also process data accroding to algorithm config
        self.aug_args = aug_args
        # save the data and label list
        self.filenames = filenames
        self.labels = labels

        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        if self.mode == 'train':
            pass
        elif self.mode == 'val':
            # data transformation
            self.data_transform = video_transforms.Compose([
            video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        elif self.mode == 'eval':
            # data transformation
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            # because it use 3 crop so no need to do centercrop
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            self.seg_crop_indexes = [(i, j) for i in range(self.eval_num_segments) for j in range(self.eval_num_crops)]
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def __getitem__(self, index):
        fn = self.filenames[index]
        label = self.labels[index]
        buffer = self.loadvideo_decord(fn) # T H W C
        if len(buffer) == 0:
            raise ValueError("Failed to load video. The video buffer is empty.")
        if self.mode == 'train':
            input_list = []
            label_list = []
            if self.aug_args.num_aug_examples < 1:
                raise ValueError("Failed to transform video. The num_aug_examples is 0.")
            for _ in range(self.aug_args.num_aug_examples):
                input_aug = self._aug_frame(buffer, self.aug_args)
                input_list.append(input_aug)
                label_list.append(label)
            return input_list, label_list
        elif self.mode == 'val':
            input_frames = self.data_transform(buffer)
            return [input_frames], [label]
        elif self.mode == 'eval':
            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)
            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                / (self.eval_num_crops - 1)
            input_list = []
            label_list = []
            file_list = []
            for i, (seg_i, crop_i) in enumerate(self.seg_crop_indexes):
                temporal_start = seg_i # 0/1
                spatial_start = int(crop_i * spatial_step)
                # crop manully to get different view of one frame
                if buffer.shape[1] >= buffer.shape[2]:
                    input_frames = buffer[temporal_start::self.eval_num_segments, \
                        spatial_start:spatial_start + self.short_side_size, :, :]
                else:
                    input_frames = buffer[temporal_start::self.eval_num_segments, \
                        :, spatial_start:spatial_start + self.short_side_size, :]

                input_frames = self.data_transform(input_frames)
                input_list.append(input_frames)
                label_list.append(label)
                file_list.append(os.path.basename(fn))
            return input_list, label_list, file_list
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, aug_args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=aug_args.aa,
            interpolation=aug_args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if aug_args.random_erase:
            erase_transform = RandomErasing(
                aug_args.reprob,
                mode=aug_args.remode,
                max_count=aug_args.recount,
                num_splits=aug_args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def loadvideo_decord(self, fname):
        """Load video content using Decord"""
        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.width, height=self.height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'eval':
            all_index = []
            average_duration = float(len(vr) - 1) / float(self.num_frames)
            interval_cross_view = average_duration / self.eval_num_segments
            for i in range(self.num_frames):
                for ci in range(self.eval_num_segments):
                    start = int(np.round(average_duration * i + interval_cross_view * ci))
                    end = int(np.round(average_duration * i + interval_cross_view * (ci + 1)))
                    all_index.append(int((start + end) // 2))
            
            all_index = list(np.array(all_index))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        else:
            # for train,and val, randomly sample frames
            all_index = []
            average_duration = float(len(vr) - 1) / self.num_frames
            for i in range(self.num_frames):
                start = int(np.round(average_duration * i))
                end = int(np.round(average_duration * (i + 1)))
                all_index.append(int(np.random.randint(start, end + 1)))

            all_index = list(np.array(all_index))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

    def __len__(self):
        return len(self.filenames)

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames

def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def build_dataset(mode, filenames, labels, dataset_config):
    dataset = SSVideoClsDataset(
        mode=mode,
        filenames = filenames,
        labels = labels,
        height=dataset_config.height,
        width=dataset_config.width,
        short_side_size=dataset_config.short_side_size,
        crop_size=dataset_config.crop_size, # equal to model input image size
        keep_aspect_ratio=dataset_config.keep_aspect_ratio,
        num_frames=dataset_config.num_frames,
        eval_num_segments=dataset_config.eval_num_segments,
        eval_num_crops=dataset_config.eval_num_crops,
        aug_args=dataset_config.aug_args
    )

    return dataset