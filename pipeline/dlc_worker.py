import argparse
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DLC worker helper")
    parser.add_argument("--action", required=True, choices=["analyze", "filter", "extract_outliers"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--destfolder", required=True)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--save-as-csv", action="store_true")
    parser.add_argument("--shuffle", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import deeplabcut

    video_p = Path(args.video).resolve()
    video_dir = video_p.parent
    destfolder = Path(args.destfolder).resolve()

    print(f"[DLC WORKER] Action={args.action} Video={video_p} Dest={destfolder}")
    cwd = os.getcwd()
    try:
        os.chdir(str(video_dir))
        if args.action == "analyze":
            kwargs = dict(
                save_as_csv=bool(args.save_as_csv),
                batchsize=int(args.batchsize),
                destfolder=str(destfolder),
            )
            if args.shuffle is not None:
                kwargs["shuffle"] = int(args.shuffle)
            deeplabcut.analyze_videos(
                args.config,
                [str(video_p)],
                **kwargs,
            )
        elif args.action == "filter":
            deeplabcut.filterpredictions(
                args.config,
                [str(video_p)],
                destfolder=str(destfolder),
            )
        elif args.action == "extract_outliers":
            deeplabcut.extract_outlier_frames(
                config=args.config,
                videos=[str(video_p)],
            )
        else:
            raise ValueError(f"Unknown action: {args.action}")
    finally:
        os.chdir(cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
