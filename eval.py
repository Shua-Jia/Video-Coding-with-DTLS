import argparse
import os
import torch
from torchvision.utils import save_image
import data_preprocess
from model.DW_Net import UNetModel
from DT_Utils import a_loss

import time
import torchvision.utils as vutils
from einops import rearrange
import torch.nn.functional as F
def run_inference(model_ckpt, output_dir):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetModel(
        encoder_params={
            "in_channels": args.in_channels,
            "initial_channels": args.initial_channels,
            "time_emb_dim": args.time_emb_dim
        },
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        dropout=args.dropout,
        decoder_params=None
    ).to(device)

    #model = UNetModel().cuda()  # 使用 .cuda() 替代 .to(device)

    if torch.cuda.is_available():
        device = f"cuda:0"
    else:
        device = torch.device("cpu")
    #if model_ckpt != 'None':
    ckpt = torch.load(model_ckpt,map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['codec'].items()})
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型的参数总量: {total_params}")
    del ckpt
    # model_en.load_state_dict(torch.load(model_en_path, map_location='cuda:0'))
    model.eval()
    psnr_dict = {}

    for index, batch_data in enumerate(dataloder):
        input_fr = batch_data["input_frames"].cuda()  # 使用 .cuda()
        gt = batch_data["gt_frames"].cuda()  # 使用 .cuda()
        start_time=time.time()

        batch_size = input_fr.size(0)
        time_emb = torch.ones(batch_size, args.time_emb_dim).to(device)
        # 前向传播
        ref_idx = args.num_frames // 2  # 参考帧索引

        with torch.no_grad():
            de_output ,diff = model(input_fr,time_emb,ref_idx)
        end_time=time.time()
        reference_time=end_time-start_time
        print(reference_time)

        b, n, c, h, w = gt.shape
       # outputs = torch.cat(de_output, dim=0)
        outputs = de_output.view(b, n, c, h, w)

        if outputs.dim() == 4:
            outputs = de_output.detach().unsqueeze(1)  # (b, 1, c, h, w)

        for b_idx in range(b):
            for n_idx in range(n):
                frame_name = "{}_{}".format(batch_data["video_name"][b_idx], batch_data["gt_names"][n_idx][b_idx])
                psnr_dict[frame_name] = a_loss.psnr_ssim.calculate_psnr(gt[b_idx, n_idx:n_idx + 1],outputs[b_idx, n_idx:n_idx + 1]).item()
                # 保存输出图像
                save_path_base = os.path.abspath(os.path.join(output_dir, batch_data["video_name"][b_idx]))
                os.makedirs(save_path_base, exist_ok=True)
                save_path = os.path.join(save_path_base, batch_data["gt_names"][n_idx][b_idx])
                img=outputs[b_idx, n_idx:n_idx + 1]
                #resized_frame = F.interpolate(img, size=(512, 512), mode='bicubic', antialias=True)

                #vutils.save_image(img.add(1).mul(0.5), save_path,normalize=True)  # antialias=True,
                vutils.save_image(img, save_path, normalize=True)  # antialias=True,

    mean_psnr = sum(psnr_dict.values()) / len(psnr_dict)


    print("Memory: ", torch.cuda.memory_allocated())
    print("mean PSNR: {:.2f} ".format(mean_psnr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FVSRs model with DWNet")

    # 数据相关参数
    parser.add_argument("--data_path", default=r"/home/user/Downloads/HEVC/ClassE_frames")
    parser.add_argument("--model_ckpt", type=str,default=r'/home/user/projects/codes/HK/lab-to-student/lab17_DT/TMM/ckpts/ckpts100_0.pth')
    parser.add_argument("--output_dir", type=str,default=r'/home/user/projects/codes/HK/lab-to-student/lab17_DT/TMM/results')

    parser.add_argument("--is_train", default=False)
    parser.add_argument("--batch_size", type=int, default=5, help="number of batch size")
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=2001)
    parser.add_argument('--gt_folder', type=str, default='sharp', help='the ground truth folder')
    parser.add_argument('--latent', type=str, default='mini_2_264', help='the input folder')
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default='None')
    # 模型相关参数
    parser.add_argument("--in_channels", type=int, default=3, help="输入通道数")
    parser.add_argument("--initial_channels", type=int, default=32, help="初始卷积后的通道数")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="时间嵌入的维度")
    parser.add_argument("--patch_size", type=int, default=4, help="补丁大小")
    parser.add_argument("--num_heads", type=int, default=8, help="头数")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout 概率")


    args = parser.parse_args()
    dataset = data_preprocess.data(args)
    dataloder = torch.utils.data.DataLoader(dataset)
    run_inference(args.model_ckpt, args.output_dir)

# 31.48  -Koj9hvcBMk_0
# 1024  33.35

# first: video_to_frame.py;  second, com_video_frame.py     third, rb.py


#  mean psnr is 36.42



#  bpp 0.000921   psnr