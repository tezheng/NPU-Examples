import argparse
from pathlib import Path


def resolve_module(filepath):
  filepath = Path(filepath).resolve()
  module_name = filepath.stem
  current_dir = filepath.parent

  # Walk up to find package root
  while current_dir:
    if not (current_dir / "__init__.py").exists():
      break
    module_name = f"{current_dir.name}.{module_name}"
    current_dir = current_dir.parent

  return module_name, current_dir


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file", type=str, help="Path to the file")
  parser.add_argument("--module-name", action='store_true', default=False,
                      help="Resolve module name")
  parser.add_argument("--package-root", action='store_true', default=False,
                      help="Resolve package root")
  args = parser.parse_args()

  if not args.file:
    raise RuntimeError("[Resolve Module] Please provide a file path!")
  if not args.module_name and not args.package_root:
    raise RuntimeError(
      "[Resolve Module] Please provide either --module-name or --package-root")

  module_name, package_root = resolve_module(args.file)
  if args.module_name:
    print(module_name)
  if args.package_root:
    print(package_root)
