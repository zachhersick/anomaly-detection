import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"


def snapshot_artifacts(directory):
    if not directory.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {directory}")
        
    snapshot = {}
    
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        
        relative_path = path.relative_to(directory).as_posix()
        
        hasher = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(chunk)
                
        snapshot[relative_path] = hasher.hexdigest()
        
    return snapshot


def main():
    before = snapshot_artifacts(ARTIFACT_DIR)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=ROOT,
        check=False,
    )
    
    try:
        after = snapshot_artifacts(ARTIFACT_DIR)
    except FileNotFoundError:
        print("\nFAIL: Artifact directory was deleted.")
        return 1
    
    added = after.keys() - before.keys()
    removed = before.keys() - after.keys()
    modified ={
        name for name in before.keys() & after.keys()
        if before[name] != after[name]
    }
    
    if added or removed or modified:
        print("\nFAIL: Production artifacts changed.")
        
        for name in sorted(added):
            print(f"  ADDED: {name}")

        for name in sorted(removed):
            print(f"  REMOVED: {name}")

        for name in sorted(modified):
            print(f"  MODIFIED: {name}")
            
    else:
        print("\nPASS: Production artifacts are byte-identical.")

    if result.returncode != 0:
        print(f"FAIL: Pytest exited with code {result.returncode}.")

    if result.returncode == 0 and not (added or removed or modified):
        print("\nAN-001 artifact integrity verification passed.")
        return 0

    return result.returncode or 1


if __name__ == '__main__':
    sys.exit(main())