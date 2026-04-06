#!/usr/bin/env python3
"""Generate Python gRPC stubs from proto files."""

import argparse
import re
import subprocess
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


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
    logger.info(f"Running: {' '.join(cmd)}")
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


def write_init_file(module_path: Path):
    """Create __init__.py exporting all messages + stub class."""
    init_path = module_path / "__init__.py"
    lines = ["# Auto-generated __init__.py\n"]

    # Step 1: Get stub class from *_pb2_grpc.py
    stub_class = None
    servicer_class = None
    for grpc_file in module_path.glob("*_pb2_grpc.py"):
        text = grpc_file.read_text()
        stub_match = re.search(r"class\s+(\w+Stub)\(", text)
        servicer_match = re.search(r"class\s+(\w+Servicer)\(", text)
        if stub_match:
            stub_class = stub_match.group(1)
        if servicer_match:
            servicer_class = servicer_match.group(1)
        if stub_class or servicer_class:
            imports = []
            if stub_class:
                imports.append(stub_class)
            if servicer_class:
                imports.append(servicer_class)
            # Also import the add_*_to_server function
            add_func_match = re.search(r"def\s+(add_\w+Servicer_to_server)", text)
            if add_func_match:
                imports.append(add_func_match.group(1))
            lines.append(f"from .{grpc_file.stem} import " + ", ".join(imports))
            break

    # Step 2: Get all message classes from *_pb2.pyi or *_pb2.py
    all_message_classes = []
    for pb2_file in module_path.glob("*_pb2.py"):
        text = pb2_file.read_text()
        # Look for DESCRIPTOR assignments which indicate message classes
        classes = re.findall(r"class\s+(\w+)\s*\(", text)
        classes = [c for c in classes if not c.startswith("_")]
        all_message_classes.extend(classes)
        if classes:
            lines.append(f"from .{pb2_file.stem} import " + ", ".join(classes))

    # Step 3: build __all__
    all_names = all_message_classes.copy()
    if stub_class:
        all_names.append(stub_class)
    if servicer_class:
        all_names.append(servicer_class)
    lines.append("\n__all__ = [\n    " + ",\n    ".join(f'"{x}"' for x in all_names) + "\n]")

    init_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate gRPC Python stubs from proto files")
    parser.add_argument("--out-dir", type=str, default="generated", help="Output directory for generated files")
    parser.add_argument("--proto-dir", type=str, default="proto", help="Directory containing proto files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    proto_dir = Path(args.proto_dir)

    out_dir.mkdir(exist_ok=True)

    # Create top-level __init__.py
    (out_dir / "__init__.py").write_text("# Auto-generated\n")

    for proto_file in proto_dir.glob("*.proto"):
        module_name = proto_file.stem
        module_out_dir = out_dir / module_name
        logger.info(f"\n=== Processing {proto_file.name} -> {module_name}/ ===")

        # Step 1: protoc
        run_protoc(proto_file, module_out_dir)

        # Step 2: fix imports
        for py_file in module_out_dir.glob("*.py"):
            fix_imports_in_file(py_file)

        # Step 3: write __init__.py
        write_init_file(module_out_dir)

        logger.info(f"Finished {proto_file.name}: __init__.py created with all messages + stub")


if __name__ == "__main__":
    main()
