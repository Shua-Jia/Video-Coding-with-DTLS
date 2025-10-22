from DT_Utils.base_args_eval import args
import torch
from einops import rearrange
import data_preprocess
from model.DW_Net import UNetModel
import os
import torchvision.utils as vutils

from thop import profile


import numpy as np
import cv2  # 假设你使用 OpenCV 来处理图像
# def save_de_out_image(de_out, video_name, index,sample_path):
#     # 保存每个图像
#
#     for i in range(de_out.size(0)):  # 遍历 batch 中的每个图像
#         image_name = f'epo_bat{index}_img{i}_{video_name[0][:8]}.png'
#         save_path = os.path.join(sample_path, image_name)
#         vutils.save_image(de_out[i], save_path, normalize=True)




def save_de_out_image(output, img_name, folder_name, output_dir):
    """
    保存深度学习模型输出的图像，使用 torchvision.utils.save_image。

    参数：
    - output: 模型输出的张量，形状为 (batch_size, channel, height, width)，值范围在 0–1。
    - img_name: 图像名称列表，形状为 (batch_size, 1)。
    - folder_name: 保存图像的文件夹名称列表。
    - output_dir: 根保存路径。
    """
    # 创建保存文件夹路径
    save_folder = os.path.join(output_dir, folder_name[0])
    os.makedirs(save_folder, exist_ok=True)

    # 检查 output 是否为 Tensor
    if not isinstance(output, torch.Tensor):
        raise TypeError("输出必须是 PyTorch 张量。")

    # 遍历每张图像并保存
    for i in range(output.shape[0]):
        # 获取当前图像名称（去掉扩展名）
        img_name_current = os.path.splitext(img_name[i][0])[0]

        # 生成完整的输出路径
        save_path = os.path.join(save_folder, f"{img_name_current}.png")

        # 保存图像，使用 normalize=True 将值归一化到 [0, 1]
        vutils.save_image(output[i], save_path, normalize=True)
        print(f"图像已保存：{save_path}")

def calculate_psnr(img1, img2):
    """
    计算PSNR（峰值信噪比）
    数据范围 [0, 1]
    """
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    # 防止除以零
    mse = torch.where(mse == 0, torch.tensor(1e-10, device=mse.device), mse)

    PIXEL_MAX = 1.0
    return 20 * torch.mean(torch.log10(PIXEL_MAX / torch.sqrt(mse)))

class evaler:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据集和数据加载器
        self.dataset = data_preprocess.data(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size)   #,shuffle=True)

        # 初始化模型
        encoder_params = {
            #"in_channels": args.in_channels,
            "initial_channels": args.initial_channels,
            "time_emb_dim": args.time_emb_dim
        }

        self.model = UNetModel(
            encoder_params=encoder_params,
            num_frames=args.num_frames,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
            decoder_params=None
        ).to(self.device)

        checkpoint = args.ckpt
        if checkpoint != 'None':
            ckpt = torch.load(checkpoint)
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['codec'].items()})
            current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
            del ckpt
    # def count_parameters(self):
    #     total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    #     print(f'Total trainable parameters: {total_params}')
    def count_parameters(self):
        print("Parameters by module:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.numel()} parameters")

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total trainable parameters: {total_params}')

    def eval(self):
        self.count_parameters()  # 打印参数量
        total_psnr = 0
        total_batches = 0
        for index, batch_data in enumerate(self.dataloader):
            # 获取输入和标签
            input_fr = batch_data["input_frames"].to(self.device)
            input = rearrange(input_fr, 'b t c n d -> (b t) c n d')
            gt = batch_data["gt_frames"].to(self.device)
            gt = rearrange(gt, 'b t c n d -> (b t) c n d')
            folder_name = batch_data['folder_name']
            img_name = batch_data['gt_names']
            # 创建时间嵌入
            batch_size = input.size(0)
            time_emb = torch.randn(batch_size, self.args.time_emb_dim).to(self.device)
            # 前向传播
            ref_idx = self.args.num_frames // 2  # 参考帧索引
            #self.model.zero_grad()
            with torch.no_grad():
                output = self.model(input, time_emb, ref_idx)
            psnr=calculate_psnr(output,gt)
            print(folder_name,'--------------', psnr)
            total_psnr += psnr
            total_batches += 1

            save_de_out_image(output, img_name, folder_name,args.output_dir)

        average_psnr = total_psnr / total_batches if total_batches > 0 else 0
        print(f'Average PSNR: {average_psnr}')

            # 主程序
if __name__ == "__main__":
    evalers = evaler(args)
    evalers.eval()


# Class E  crf23   32.17(6 video)   32.574(3 video, crf23)   34.38
#  Class D  crf23

# 30.31  CRF 23    crf 0  32.2999   # 24.29 Spec   24.33   24.90  # 32.89   32.71





















































































































































































































































































