
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


        for frame_name in gt_frames_name:
            # 打开图像
            img = Image.open(gt_frames_path.format(frame_name))
            width0, height0 = img.size
            if self.args.is_train:
                left = random.randint(0, img.width - crop_width)
                upper = random.randint(0, img.height - crop_height)
                img = img.crop((left, upper, left + crop_width, upper + crop_height))
            img_tensor = self.transform(img)

            gt_frames.append(img_tensor)

        #channels, height, width=img_tensor.size()


        for frame_name in input_frames_name:
            # 打开图像
            img = Image.open(input_frames_path.format(frame_name))
            img = self.totensor(img)
            #width, height = img.size
            img = resize_frame(img, height0, width0)
            #img=F.interpolate(img.unsqueeze(0), size=(height0, width0), mode='bicubic', antialias=True).squeeze(0)
            if self.args.is_train:
                img = img[:, upper:upper + crop_height, left:left + crop_width]
            # 转换为张量
            img_tensor = self.transform0(img)
            #img_tensor=resize_frame(img_tensor,height, width)
            # 将结果添加到 input_frames 列表
            input_frames.append(img_tensor)


        input_frames = torch.stack(input_frames)
        gt_frames = torch.stack(gt_frames)
        folder_name = input_frames_path.split('/')[-3]


        return {"input_frames": input_frames,"gt_frames": gt_frames, "video_name": video_name,"video_length": video_length, "gt_names": gt_frames_name,'folder_name':folder_name, }

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



def resize_frame(frame, height, width):
    if height <= 1022 and width <= 1022:
        # For 512x512, choose one of three resizing paths with equal probability

        choice0 = random.choices(
            [1, 2, 3],  # 1: stop at height//4, 2: resize to height//8, 3: resize to height//16
            weights=[2 / 10, 3.5 / 10, 4.5 / 10],
        )
        if choice0 == 1:
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
        elif choice0 == 2:
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 4, width // 4), mode='bicubic', antialias=True)
        else:  # choice == 3
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 4, width // 4), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 8, width // 8), mode='bicubic', antialias=True)
    elif height == 256 and width == 256:
        # For 256x256, choose one of two resizing paths with equal probability
        choice1 = random.choice([1, 2])
        if choice1 == 1:
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
        else:  # choice == 2
            frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 4, width // 4), mode='bicubic', antialias=True)
    else:
        # Original logic for dimensions > 512
        frame = F.interpolate(frame.unsqueeze(0), size=(height // 2, width // 2), mode='bicubic', antialias=True)
        frame = F.interpolate(frame, size=(height // 4, width // 4), mode='bicubic', antialias=True)
        choice2 = random.choices(
            [1, 2, 3],  # 1: stop at height//4, 2: resize to height//8, 3: resize to height//16
            weights=[3 / 10, 3/ 10, 4 / 10],
            k=1
        )[0]
        if choice2 == 2:
            frame = F.interpolate(frame, size=(height // 8, width // 8), mode='bicubic', antialias=True)
        elif choice2 == 3:
            frame = F.interpolate(frame, size=(height // 8, width // 8), mode='bicubic', antialias=True)
            frame = F.interpolate(frame, size=(height // 16, width // 16), mode='bicubic', antialias=True)

    # Upscale back to original size
    resized_frame = F.interpolate(frame, size=(height, width), mode='bicubic', antialias=True)
    return resized_frame.squeeze(0)
