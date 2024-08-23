import argparse
import math
import os
from sys import stderr

import torch
from torch.utils.data import DataLoader, IterableDataset

try:
    import pillow_jxl
except ImportError:
    pass

from model import ClassifierModel, BACKBONE_RES, QUALITY_RES
from image import load_image

class ImageDataset(IterableDataset):
    def __init__(self, path: str, *, file_list: str | None = None, recurse: bool = False):
        self.base = None
        self.paths = []

        if file_list is not None:
            self.base = path
            with open(file_list, "r", encoding="utf8") as file:
                self.paths = [line.rstrip() for line in file if line.rstrip()]
        elif os.path.isdir(path):
            self.base = path
            if recurse:
                self.paths = [
                    os.path.join(subdir, filename)
                    for subdir, _, filenames in os.walk(path)
                    for filename in filenames
                    if not filename.startswith('.')
                ]
            else:
                self.paths = [
                    os.path.join(path, filename)
                    for filename in os.listdir(path)
                    if not filename.startswith('.') and os.path.isfile(os.path.join(path, filename))
                ]
        else:
            self.base = None
            self.paths = [path]

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.load_images(0, len(self.paths))

        per_worker = int(math.ceil(len(self.paths) / float(worker_info.num_workers)))
        start = worker_info.id * per_worker
        return self._load_images(start, min(start + per_worker, len(self.paths)))

    def _load_images(self, start: int, end: int):
        for idx in range(start, end):
            path = self.paths[idx]
            full_path = path if self.base is None else os.path.join(self.base, path)

            try:
                xb, xq = load_image(full_path, BACKBONE_RES, QUALITY_RES)
            except Exception as ex:
                print(f"WARNING: \"{path}\": {ex}", file=stderr)
                continue

            yield path, torch.from_numpy(xb), torch.from_numpy(xq)

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Assess images with FARAMIR.")
    parser.add_argument("path", type=str, help="Directory of image files to classify, or a path to a single image file.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("-l", "--file_list", type=str, help="Use the specified list of files relative to path, instead of enumerating all files.")
    parser.add_argument("-m", "--model", type=str, default="faramir-v1.pth", help="Path to the FARAMIR checkpoint.")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="PyTorch inference device.")
    parser.add_argument("-c", "--compile_mode", type=str, default="auto", help="Compilation mode for PyTorch, or 'off' or 'auto'.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Confidence threshold for acceptance.")
    parser.add_argument("-r", "--recurse", action="store_true", help="Recursively process the specified directory.")
    args = parser.parse_args()

    dataset = ImageDataset(
        args.path,
        file_list=args.file_list,
        recurse=args.recurse
    )

    if len(dataset) == 0:
        raise RuntimeError("no files")

    model = ClassifierModel()

    _, unexpected = model.load_state_dict(torch.load(args.model, weights_only=True), strict=False)
    if len(unexpected) != 0:
        raise RuntimeError(f"\"{args.model}\" contains unexpected weights: {unexpected}")

    model.eval().to(args.device)

    if args.device == "cpu":
        args.compile_mode = "off"
    elif args.compile_mode == "auto":
        args.compile_mode = "max-autotune" if (len(dataset) / args.batch_size) >= 64 else "off"

    compiled_model = None
    if args.compile_mode != "off":
        compiled_model = torch.compile(model, mode=args.compile_mode)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        prefetch_factor=math.ceil(args.batch_size / os.cpu_count()),
        pin_memory=(args.device != "cpu")
    )

    with torch.no_grad():
        for paths, xb, xq in dataloader:
            xb = xb.to(device=args.device, dtype=torch.float32, memory_format=torch.channels_last, non_blocking=True)
            xq = xq.to(device=args.device, dtype=torch.float32, memory_format=torch.channels_last, non_blocking=True)

            if compiled_model is not None and (
                len(xb) == args.batch_size or
                args.compile_mode == "reduce-overhead"
            ):
                outputs = compiled_model(xb, xq)
            else:
                outputs = model(xb, xq)

            assessments = model.assess(outputs, args.threshold)
            for path, (accept, cls, confidence) in zip(paths, assessments):
                status = "accept" if accept else "reject"
                print(f"{path}\t{status}\t{cls}\t{confidence:.2f}")

if __name__ == "__main__":
    main()
