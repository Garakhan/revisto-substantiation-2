#!/usr/bin/env python3
"""Generate gRPC stubs for the NLP client."""

import re
import subprocess
from pathlib import Path


def run_protoc(proto_path: Path, out_path: Path):
    """Run protoc to generate Python + gRPC stubs."""
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={proto_path.parent}",
        f"--python_out={out_path}",
        f"--grpc_python_out={out_path}",
        str(proto_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def fix_imports_in_file(file_path: Path):
    """Fix generated imports to be relative."""
    text = file_path.read_text()
    text = re.sub(
        r"import (\w+_pb2) as (\w+__pb2)",
        r"from . import \1 as \2",
        text,
    )
    file_path.write_text(text)


def write_init_file(module_path: Path):
    """Create __init__.py exporting stub class and messages."""
    init_path = module_path / "__init__.py"
    lines = ["# Auto-generated __init__.py\n"]

    # Get stub class from *_pb2_grpc.py
    for grpc_file in module_path.glob("*_pb2_grpc.py"):
        text = grpc_file.read_text()
        stub_match = re.search(r"class\s+(\w+Stub)\(", text)
        if stub_match:
            lines.append(f"from .{grpc_file.stem} import {stub_match.group(1)}")
            break

    # Get message classes from *_pb2.py
    for pb2_file in module_path.glob("*_pb2.py"):
        text = pb2_file.read_text()
        classes = re.findall(r"class\s+(\w+)\s*\(", text)
        classes = [c for c in classes if not c.startswith("_")]
        if classes:
            lines.append(f"from .{pb2_file.stem} import " + ", ".join(classes))

    init_path.write_text("\n".join(lines))


def main():
    script_dir = Path(__file__).parent
    proto_dir = script_dir / "proto"
    out_dir = script_dir / "generated"

    out_dir.mkdir(exist_ok=True)
    (out_dir / "__init__.py").write_text("# Auto-generated\n")

    for proto_file in proto_dir.glob("*.proto"):
        module_name = proto_file.stem
        module_out_dir = out_dir / module_name
        print(f"\n=== Processing {proto_file.name} -> {module_name}/ ===")

        # Generate stubs
        run_protoc(proto_file, module_out_dir)

        # Fix imports
        for py_file in module_out_dir.glob("*.py"):
            fix_imports_in_file(py_file)

        # Write __init__.py
        write_init_file(module_out_dir)

        print(f"Finished {proto_file.name}")


if __name__ == "__main__":
    main()
