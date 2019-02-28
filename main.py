import os
from SRNTT.model import *
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='SRNTT')

# init parameters
parser.add_argument('--is_train', type=str2bool, default=False)
parser.add_argument('--srntt_model_path', type=str, default='SRNTT/models/SRNTT')
parser.add_argument('--vgg19_model_path', type=str, default='SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat')
parser.add_argument('--save_dir', type=str, default=None, help='dir of saving intermediate training results')
parser.add_argument('--num_res_blocks', type=int, default=16, help='number of residual blocks')

# train parameters
parser.add_argument('--input_dir', type=str, default='data/train/input', help='dir of input images')
parser.add_argument('--ref_dir', type=str, default='data/train/ref', help='dir of reference images')
parser.add_argument('--map_dir', type=str, default='data/train/map_321', help='dir of texture maps of reference images')
parser.add_argument('--batch_size', type=int, default=9)
parser.add_argument('--num_init_epochs', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--use_pretrained_model', type=str2bool, default=True)
parser.add_argument('--use_init_model_only', type=str2bool, default=False, help='effect if use_pretrained_model is true')
parser.add_argument('--w_per', type=float, default=1e-4, help='weight of perceptual loss between output and ground truth')
parser.add_argument('--w_tex', type=float, default=1e-4, help='weight of texture loss between output and texture map')
parser.add_argument('--w_adv', type=float, default=1e-6, help='weight of adversarial loss')
parser.add_argument('--w_bp', type=float, default=0.0, help='weight of back projection loss')
parser.add_argument('--w_rec', type=float, default=1.0, help='weight of reconstruction loss')
parser.add_argument('--vgg_perceptual_loss_layer', type=str, default='relu5_1', help='the VGG19 layer name to compute perceptrual loss')
parser.add_argument('--is_WGAN_GP', type=str2bool, default=True, help='whether use WGAN-GP')
parser.add_argument('--is_L1_loss', type=str2bool, default=True, help='whether use L1 norm')
parser.add_argument('--param_WGAN_GP', type=float, default=10, help='parameter for WGAN-GP')
parser.add_argument('--input_size', type=int, default=40)
parser.add_argument('--use_weight_map', type=str2bool, default=False)
parser.add_argument('--use_lower_layers_in_per_loss', type=str2bool, default=False)

# test parameters
parser.add_argument('--result_dir', type=str, default='result', help='dir of saving testing results')
parser.add_argument('--ref_scale', type=float, default=1.0)
parser.add_argument('--is_original_image', type=str2bool, default=True)

args = parser.parse_args()

if args.is_train:

    # record parameters to file
    if args.save_dir is None:
        args.save_dir = 'default_save_dir'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'arguments.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            line = '{:>30}\t{:<10}\n'.format(arg, getattr(args, arg))
            bar = ''
            f.write(line)
        f.close()

    srntt = SRNTT(
        srntt_model_path=args.srntt_model_path,
        vgg19_model_path=args.vgg19_model_path,
        save_dir=args.save_dir,
        num_res_blocks=args.num_res_blocks
    )
    srntt.train(
        input_dir=args.input_dir,
        ref_dir=args.ref_dir,
        map_dir=args.map_dir,
        batch_size=args.batch_size,
        num_init_epochs=args.num_init_epochs,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        use_pretrained_model=args.use_pretrained_model,
        use_init_model_only=args.use_init_model_only,
        weights=(args.w_per, args.w_tex, args.w_adv, args.w_bp, args.w_rec),
        vgg_perceptual_loss_layer=args.vgg_perceptual_loss_layer,
        is_WGAN_GP=args.is_WGAN_GP,
        is_L1_loss=args.is_L1_loss,
        param_WGAN_GP=args.param_WGAN_GP,
        input_size=args.input_size,
        use_weight_map=args.use_weight_map,
        use_lower_layers_in_per_loss=args.use_lower_layers_in_per_loss
    )
else:
    if args.save_dir is not None:
        # read recorded arguments
        fixed_arguments = ['srntt_model_path', 'vgg19_model_path', 'save_dir', 'num_res_blocks', 'use_weight_map']
        if os.path.exists(os.path.join(args.save_dir, 'arguments.txt')):
            with open(os.path.join(args.save_dir, 'arguments.txt'), 'r') as f:
                for arg, line in zip(sorted(vars(args)), f.readlines()):
                    arg_name, arg_value = line.strip().split('\t')
                    if arg_name in fixed_arguments:
                        fixed_arguments.remove(arg_name)
                        try:
                            if isinstance(getattr(args, arg_name), bool):
                                setattr(args, arg_name, str2bool(arg_value))
                            else:
                                setattr(args, arg_name, type(getattr(args, arg_name))(arg_value))
                        except:
                            print('Unmatched arg_name: %s!' % arg_name)

    srntt = SRNTT(
        srntt_model_path=args.srntt_model_path,
        vgg19_model_path=args.vgg19_model_path,
        save_dir=args.save_dir,
        num_res_blocks=args.num_res_blocks,
    )

    srntt.test(
        input_dir=args.input_dir,
        ref_dir=args.ref_dir,
        use_pretrained_model=args.use_pretrained_model,
        use_init_model_only=args.use_init_model_only,
        use_weight_map=args.use_weight_map,
        result_dir=args.result_dir,
        ref_scale=args.ref_scale,
        is_original_image=args.is_original_image
    )

