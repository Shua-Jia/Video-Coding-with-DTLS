import platform
import torch
import os
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random

class data(torch.utils.data.Dataset):
    def __init__(self,args):
        self.args=args
        self.video_list = os.listdir(self.args.data_path)
        self.video_list.sort()

        self.frames = []
        self.video_frame_dict = {}
        self.video_length_dict = {}

        for video_name in self.video_list:

            video_path = os.path.join(self.args.data_path, video_name, self.args.gt_folder)
            frames_in_video = os.listdir(video_path)
            frames_in_video.sort()

            frames_in_video = [os.path.join(video_name, frame) for frame in frames_in_video]

            # sample length with inerval
            sampled_frames_length = (args.num_frames - 1) * args.interval + 1

            self.frames += frames_in_video[sampled_frames_length // 2: len(frames_in_video) - (sampled_frames_length // 2)]

            self.video_frame_dict[video_name] = frames_in_video
            self.video_length_dict[video_name] = len(frames_in_video)
            self.totensor=transforms.ToTensor()


            if self.args.is_train:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Resize(256),
                    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

                ])
            if self.args.is_train:
                self.transform0 = transforms.Compose([
                    #transforms.ToTensor(),
                    #transforms.Resize(256),
                    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])
            else:
                self.transform0 = transforms.Compose([
                    #transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

                ])


    def __getitem__(self, idx):
        if platform.system() == "Windows":
            video_name, frame_name = self.frames[idx].split("\\")
        else:
            video_name, frame_name = self.frames[idx].split("/")
        frame_idx, suffix = frame_name.split(".")
        frame_idx = int(frame_idx)
        video_length = self.video_length_dict[video_name]
        gt_frames_name = [frame_name]
        # when to read the frames, should pay attention to the name of frames
        input_frames_name = ["{:06d}.{}".format(i, suffix) for i in range(
                frame_idx - (self.args.num_frames // 2) * self.args.interval, frame_idx + (self.args.num_frames // 2) * self.args.interval + 1, self.args.interval)]

        assert len(input_frames_name) == self.args.num_frames, "Wrong frames length not equal the sampling frames {}".format(
            self.args.num_frames)

        gt_frames_path = os.path.join(self.args.data_path, video_name, self.args.gt_folder, "{}")
        input_frames_path = os.path.join(self.args.data_path, video_name, self.args.latent, "{}")

        gt_frames = []
        input_frames=[]

        crop_width = 320
        crop_height = 192

        # 读取 GT，得到 height0/width0，并进行可选裁剪与归一化
        for frame_name in gt_frames_name:
            img = Image.open(gt_frames_path.format(frame_name))
            width0, height0 = img.size
            if self.args.is_train:
                left = random.randint(0, img.width - crop_width)
                upper = random.randint(0, img.height - crop_height)
                img = img.crop((left, upper, left + crop_width, upper + crop_height))
            img_tensor = self.transform(img)
            gt_frames.append(img_tensor)

        # 为本样本的所有 input 帧统一一次缩放策略 choice（根据尺寸分支选择集合）
        if self.args.is_train:
            # 训练阶段：随机选择一个统一的 choice
            if height0 <= 1022 and width0 <= 1022:
                choice_unified = random.choices([1, 2, 3], weights=[2/10, 3.5/10, 4.5/10], k=1)[0]
            elif height0 == 256 and width0 == 256:
                choice_unified = random.choice([1, 2])
            else:
                choice_unified = random.choices([1, 2, 3], weights=[3/10, 3/10, 4/10], k=1)[0]
        else:
            # 测试/验证阶段：如需固定，可设为 1；也可按训练同分布，这里采用固定为 1，保证确定性
            choice_unified = 1

        # 处理 input 帧：统一使用 choice_unified
        for frame_name in input_frames_name:
            img = Image.open(input_frames_path.format(frame_name))
            img = self.totensor(img)
            img = resize_frame(img, height0, width0, choice=choice_unified)
            if self.args.is_train:
                img = img[:, upper:upper + crop_height, left:left + crop_width]
            img_tensor = self.transform0(img)
            input_frames.append(img_tensor)

        input_frames = torch.stack(input_frames)
        gt_frames = torch.stack(gt_frames)
        folder_name = input_frames_path.split('/')[-3]

        return {
            "input_frames": input_frames,
            "gt_frames": gt_frames,
            "video_name": video_name,
            "video_length": video_length,
            "gt_names": gt_frames_name,
            'folder_name': folder_name,
        }

    def __len__(self):
        return len(self.frames)


import torch.nn.functional as F

# def resize_frame(frame,height,width):
#
#     # frame = F.interpolate(frame.unsqueeze(0), size=(height//2, width//2), mode='bicubic', antialias=True)
#     # frame = F.interpolate(frame, size=(height//4, width//4), mode='bicubic', antialias=True)
#     #
#     # frame = F.interpolate(frame, size=(height//8, width//8), mode='bicubic', antialias=True)
#     # resized_frame = F.interpolate(frame, size=(height, width), mode='bicubic', antialias=True)
#
#     resized_frame = F.interpolate(frame.unsqueeze(0), size=(height, width), mode='bicubic', antialias=True)
#
#         # 压缩为 32x32
#     return resized_frame.squeeze(0)


def resize_frame(frame, height, width, choice=None):
    """
    统一/可控的多步缩放，再上采样回原尺寸。
    当 choice 为 None 时，按原逻辑随机；否则按传入 choice 执行。
    choice 取值：
      - 若 height<=1022 且 width<=1022：1/2/3
      - 若 height==256 且 width==256：1/2
      - 其他情况：1/2/3
    """
    if height <= 1022 and width <= 1022:
        # 512~1022 范围逻辑
        if choice is None:
            choice0 = random.choices(
                [1, 2, 3],
                weights=[2 / 10, 3.5 / 10, 4.5 / 10],
                k=1
            )[0]
        else:
            choice0 = choice

        if choice0 == 1:
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
        elif choice0 == 2:
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 4, width // 4), mode='bicubic', antialias=True)
        else:  # 3
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 4, width // 4), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 8, width // 8), mode='bicubic', antialias=True)

    elif height == 256 and width == 256:
        # 256x256 逻辑
        if choice is None:
            choice1 = random.choice([1, 2])
        else:
            choice1 = choice

        if choice1 == 1:
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
        else:  # 2
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 4, width // 4), mode='bicubic', antialias=True)
    else:
        # 原 > 1022 或其他尺寸逻辑：先到 1/4，再决定是否到 1/8 或 1/16
        frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
        frame = F.interpolate(frame, size=(height // 4, width // 4), mode='bicubic', antialias=True)

        if choice is None:
            choice2 = random.choices(
                [1, 2, 3],
                weights=[3 / 10, 3 / 10, 4 / 10],
                k=1
            )[0]
        else:
            choice2 = choice

        if choice2 == 2:
            frame = F.interpolate(frame, size=(height // 8, width // 8), mode='bicubic', antialias=True)
        elif choice2 == 3:
            frame = F.interpolate(frame, size=(height // 8, width // 8), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 16, width // 16), mode='bicubic', antialias=True)

    # Upscale back to original size
    resized_frame = F.interpolate(frame, size=(height, width), mode='bicubic', antialias=True)
    return resized_frame.squeeze(0)