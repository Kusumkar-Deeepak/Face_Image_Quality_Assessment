"""Command-line options shared by main.py and batch_evaluate.py."""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from face_detection import DetectorConfig
from quality_pipeline import QualityPipeline, QualityThresholds


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--variant', default='s', choices=['t', 's', 'm', 'l'], help='eDifFIQA variant (T weights are not redistributed upstream)')
    parser.add_argument('--weights', default=None, help='Path to .pth state dict (default weights/ediffiqa_<variant>.pth)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Torch device for eDifFIQA')
    parser.add_argument('--aligned', action='store_true', help='Skip detection; inputs must be aligned 112x112 BGR/RGB-file face crops')
    parser.add_argument('--detector', choices=['scrfd', 'retinaface'], default=None, help='Primary detector (default scrfd)')
    parser.add_argument('--fallback-detector', choices=['scrfd', 'retinaface'], default=None, help='Run only when the primary yields no usable face')
    parser.add_argument('--detector-config', default=None, help='JSON detector configuration (see detector_config.json)')
    parser.add_argument('--scrfd-input-size', type=int, default=None, help='Square SCRFD input size, e.g. 320 (UniFace default 640)')
    parser.add_argument('--scrfd-confidence', type=float, default=None, help='SCRFD confidence threshold (UniFace default 0.5; change only with evidence)')
    parser.add_argument('--retinaface-input-size', type=int, default=None)
    parser.add_argument('--retinaface-confidence', type=float, default=None)
    parser.add_argument('--thresholds', default=None, help='JSON quality/pose thresholds; null values leave decisions NOT_EVALUATED')
    parser.add_argument('--head-pose', action='store_true', help='Estimate pitch/yaw/roll with UniFace HeadPose')


def pipeline_from_args(args: argparse.Namespace) -> QualityPipeline:
    config = DetectorConfig.from_file(args.detector_config)
    scrfd, retina = config.scrfd, config.retinaface
    if args.scrfd_input_size is not None:
        scrfd = replace(scrfd, input_size=(args.scrfd_input_size, args.scrfd_input_size))
    if args.scrfd_confidence is not None:
        scrfd = replace(scrfd, confidence_threshold=args.scrfd_confidence)
    if args.retinaface_input_size is not None:
        retina = replace(retina, input_size=(args.retinaface_input_size, args.retinaface_input_size))
    if args.retinaface_confidence is not None:
        retina = replace(retina, confidence_threshold=args.retinaface_confidence)
    config = replace(config, scrfd=scrfd, retinaface=retina)
    return QualityPipeline(
        variant=args.variant,
        weights=args.weights,
        device=args.device,
        detector=args.detector,
        fallback_detector=args.fallback_detector,
        detector_config=config,
        thresholds=QualityThresholds.from_file(args.thresholds),
        head_pose=args.head_pose,
    )
