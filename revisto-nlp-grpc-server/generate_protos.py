import argparse
import re
import subprocess
from pathlib import Path


def run_protoc(proto_path: Path, out_path: Path):
    """Run protoc to generate Python + gRPC + mypy stubs."""
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={proto_path.parent}",
        f"--python_out={out_path}",
        f"--grpc_python_out={out_path}",
        f"--mypy_out={out_path}",
        f"--mypy_grpc_out={out_path}",
        str(proto_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fix_imports_in_file(file_path: Path):
    """Fix generated imports to be relative to the generated module."""
    text = file_path.read_text()
    # Fix pb2 imports
    text = re.sub(
        r"import (\w+_pb2) as (\w+__pb2)",
        r"from . import \1 as \2",
        text,
    )
    file_path.write_text(text)


def extract_classes_from_pyi(pyi_file: Path):
    """Extract all class names from a .pyi stub file."""
    text = pyi_file.read_text()
    classes = re.findall(r"class\s+([A-Za-z0-9_]+)\s*\(", text)
    # Exclude private/internal classes
    return [c for c in classes if not c.startswith("_")]


def extract_stub_class_from_pyi(grpc_pyi_file: Path):
    """Find the stub class in _pb2_grpc.pyi (ends with 'Stub')."""
    classes = extract_classes_from_pyi(grpc_pyi_file)
    for cls in classes:
        if cls.endswith("Stub"):
            return cls
    return None


def write_init_file(module_path: Path):
    """Create __init__.py exporting all messages + stub class."""
    init_path = module_path / "__init__.py"
    lines = ["# Auto-generated __init__.py\n"]

    # Step 1: Get stub class from *_pb2_grpc.py
    stub_class = None
    for grpc_file in module_path.glob("*_pb2_grpc.py"):
        text = grpc_file.read_text()
        match = re.search(r"class\s+(\w+Stub)\(", text)
        if match:
            stub_class = match.group(1)
            lines.append(f"from .{grpc_file.stem} import {stub_class}")
            break

    # Step 2: Get all message classes from *_pb2.pyi
    all_message_classes = []
    for pb2_pyi_file in module_path.glob("*_pb2.pyi"):
        text = pb2_pyi_file.read_text()
        classes = re.findall(r"class\s+(\w+)\s*\(", text)
        classes = [c for c in classes if not c.startswith("_")]
        all_message_classes.extend(classes)
        if classes:
            lines.append(f"from .{pb2_pyi_file.stem.replace('.pyi', '')} import " + ", ".join(classes))

    # Step 3: build __all__
    all_names = all_message_classes + ([stub_class] if stub_class else [])
    lines.append("\n__all__ = [\n    " + ",\n    ".join(f'"{x}"' for x in all_names) + "\n]")

    init_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Fix protoc imports to use a top-level package")
    parser.add_argument("out_dir", type=str, help="Root folder of generated Python files")
    parser.add_argument("proto_dir", type=str, help="Root folder of generated Python files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    proto_dir = Path(args.proto_dir)

    out_dir.mkdir(exist_ok=True)

    for proto_file in proto_dir.glob("*.proto"):
        module_name = proto_file.stem
        module_out_dir = out_dir / module_name
        print(f"\n=== Processing {proto_file.name} -> {module_name}/ ===")

        # Step 1: protoc
        run_protoc(proto_file, module_out_dir)

        # Step 2: fix imports
        for py_file in module_out_dir.glob("*.py"):
            fix_imports_in_file(py_file)

        # Step 3: write __init__.py using .pyi stubs
        write_init_file(module_out_dir)

        print(f"✓ Finished {proto_file.name}: __init__.py created with all messages + stub")


if __name__ == "__main__":
    main()
