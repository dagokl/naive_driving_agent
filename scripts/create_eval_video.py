import argparse
import re
from pathlib import Path

import cv2
from tqdm import tqdm


def get_route_num(path: Path):
    m = re.search(r'route(\d+)', path.name)
    if m:
        return int(m.group(1))
    return None


def generate_eval_video(result_dir: Path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter((result_dir / 'video.mp4').as_posix(), fourcc, 20, (2560, 1440))

    route_dirs = sorted(
        [p for p in result_dir.iterdir() if p.is_dir() and get_route_num(p) is not None],
        key=get_route_num
    )
    for route_dir in tqdm(route_dirs):
        img_dirs = (route_dir / f'images/input', route_dir / f'images/chase', route_dir / f'plots')
        i = 0
        while True:
            imgs = []
            for img_dir in img_dirs:
                img_path_jpg = img_dir / f'{i:06}.jpg'
                img_path_png = img_dir / f'{i:06}.png'

                if img_path_jpg.exists():
                    img_path = img_path_jpg
                elif img_path_png.exists():
                    img_path = img_path_png
                else:
                    break
                imgs.append(cv2.imread(img_path.as_posix()))

            if len(imgs) != 3:
                break

            imgs[0] = cv2.resize(imgs[0], (imgs[1].shape[1], imgs[1].shape[0]))

            vertical_stack = cv2.vconcat(imgs[:2])
            final_image = cv2.hconcat((vertical_stack, imgs[2]))
            video.write(final_image)
            i += 1

    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from a evaluation run.')
    parser.add_argument('result_dir', type=str, help='Path to the result dir.')
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    generate_eval_video(result_dir)
