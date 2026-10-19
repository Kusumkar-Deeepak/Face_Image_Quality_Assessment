# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse

from pipeline_cli import add_pipeline_arguments, pipeline_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='eDifFIQA PyTorch Face Quality Inference - Score one or more images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('images', type=str, nargs='+', help='One or more image paths')
    add_pipeline_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = pipeline_from_args(args)
    fallback = pipeline.fallback_detector_name or 'none'
    print(f'Loading eDifFIQA-{args.variant.upper()} on {args.device}; detector={pipeline.detector_name} fallback={fallback}')
    print(f'{"image":<40} {"detector":<22} {"status":<16} {"score":>10} {"decision":<15}')
    print('-' * 110)

    for image_path in args.images:
        for result in pipeline.evaluate(image_path, aligned=args.aligned):
            score = '-' if result.quality_score is None else f'{result.quality_score:.4f}'
            print(
                f'{result.image:<40} {result.detector:<22} {result.status:<16} '
                f'{score:>10} {result.decision:<15}'
            )
            if result.error_reason:
                print(f'  reason: {result.error_reason}')
            if result.decision_reason and result.status == 'OK':
                print(f'  decision reason: {result.decision_reason}')
            if result.image_width is not None:
                print(f'  dimensions: {result.image_width}x{result.image_height}')
            if result.face_index is not None and not args.aligned:
                confidence = '-' if result.confidence is None else f'{result.confidence:.6f}'
                print(f'  face={result.face_index} confidence={confidence} bbox={result.bbox} face_area_ratio={result.face_area_ratio:.2f}')
            if result.pose_status == 'OK':
                print(f'  head pose (deg): pitch={result.pitch:.1f} yaw={result.yaw:.1f} roll={result.roll:.1f}')
            elif result.pose_status == 'ERROR':
                print('  head pose: ERROR (quality score unaffected)')
            print(f'  time: {result.processing_time_ms:.0f} ms')


if __name__ == '__main__':
    main()
