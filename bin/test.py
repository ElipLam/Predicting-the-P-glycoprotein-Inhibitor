import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training Viet Nam traffic sign', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bs', '--batch_size', type=int,
                        help='batch size', default=128)
    parser.add_argument('--isize', '--image_size', nargs=2,
                        type=int, help='image size', default=(224, 224))
    parser.add_argument('--epos', '--epochs', type=int,
                        help='epochs', default=2)
    parser.add_argument(
        '--model', type=str, choices=['keras', 'simple', 'vgg16', 'mobilenetv2'], default='mobilenetv2')
    args = parser.parse_args()
    print(args.bs)
    print((args.isize))
    print(args.epos)
    print(args.model)
