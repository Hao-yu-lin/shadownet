import argparse

def parse_args():
    desc = "Pytorch implementation of Shadowremoval"
    parser = argparse.ArgumentParser(description = desc)
     
    parser.add_argument('--train_path', type=str, default='/home/haoyu/Desktop/partical/ShadowNet_Data/train5', help='train_path')
    parser.add_argument('--test_path', type=str, default='/home/haoyu/Desktop/partical/ShadowNet_Data/test', help='test_path')
    parser.add_argument('--result_dir', type=str, default='/home/haoyu/Desktop/partical/shadow_removal3/output/img')
    parser.add_argument('--ckpt_dir', type=str, default='/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt')
    
    # pretrain
    parser.add_argument('--pre_trained', type=bool, default=False)
    parser.add_argument('--prepretrain_path', type=str, default="/home/haoyu/Desktop/GUI/parial-GUI/pycopy/module_ckpt/shadow_removal.pt")
    
    parser.add_argument('--iteration', type=int, default=78080, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=100, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--g_update', type=int, default=2, help='The number of G update loss')
    parser.add_argument('--d_update', type=int, default=2, help='The number of D update loss')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=15, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=20, help='Weight for Identity')
    parser.add_argument('--dom_weight', type=int, default=2, help='Weight for domain classification')
    parser.add_argument('--ill_weight', type=int, default=5, help='Weight for illuminant')
    
    parser.add_argument('--ch', type=int, default=32, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
    
    parser.add_argument('--img_size', type=int, default=128, help='The size of image')
    parser.add_argument('--img_h', type=int, default=256, help='The org size of image')
    parser.add_argument('--img_w', type=int, default=256, help='The org size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--norm_type', type=str, default='norm2', help='train_path')
    
    
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--seed', type=int, default=999)
    
   
     
    args = parser.parse_args(args=[])
    
    return args

if __name__ == '__main__':
    args = parse_args()
    