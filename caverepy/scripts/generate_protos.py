"""Regenerate Python gRPC stubs from simulation.proto."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROTO_DIR = REPO_ROOT / "src" / "Grpc" / "Protos"
PROTO_FILE = PROTO_DIR / "simulation.proto"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "cavere" / "_generated"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={OUTPUT_DIR}",
        f"--grpc_python_out={OUTPUT_DIR}",
        str(PROTO_FILE),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("Proto generation failed!", file=sys.stderr)
        sys.exit(1)

    # Fix the import in the generated grpc file to use relative imports
    grpc_file = OUTPUT_DIR / "simulation_pb2_grpc.py"
    if grpc_file.exists():
        content = grpc_file.read_text()
        content = content.replace(
            "import simulation_pb2 as simulation__pb2",
            "from cavere._generated import simulation_pb2 as simulation__pb2",
        )
        grpc_file.write_text(content)
        print("Fixed imports in simulation_pb2_grpc.py")

    print("Proto generation complete!")


if __name__ == "__main__":
    main()
