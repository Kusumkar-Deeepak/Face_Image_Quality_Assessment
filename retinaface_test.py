import argparse

import cv2
import numpy as np
import torch
from torchvision import transforms

from models import get_model
from uniface import face_alignment
from uniface.detection import RetinaFace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test RetinaFace detection with eDifFIQA scoring')
    parser.add_argument('images', type=str, nargs='+', help='One or more image paths')
    parser.add_argument('--variant', type=str, default='s', choices=['t', 's', 'm', 'l'])
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    weights_path = args.weights or f'weights/ediffiqa_{args.variant}.pth'

    print(f'Loading eDifFIQA-{args.variant.upper()} from {weights_path} on {args.device}')
    model = get_model(args.variant, return_feat=False)
    state_dict = torch.load(weights_path, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(args.device).eval()

    transform = build_transform()
    detector = RetinaFace()

    for image_path in args.images:
        image = cv2.imread(image_path)
        if image is None:
            print(f'{image_path}: READ_ERR')
            continue

        faces = detector.detect(image)
        print(f'{image_path}: detected_faces={len(faces)}')

        for index, face in enumerate(faces, start=1):
            print(f'  face={index} confidence={face.confidence:.6f} bbox={face.bbox.tolist()}')
            aligned, _ = face_alignment(image, face.landmarks.astype(np.float32))
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(args.device)
            score = model(tensor).squeeze().detach().cpu().item()
            print(f'  face={index} quality_score={score:.4f}')


if __name__ == '__main__':
    main()