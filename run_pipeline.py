import subprocess
import sys


PIPELINE_SCRIPTS = [
    "generator.py",
    "features.py",
    "model.py",
    "evaluate.py",
    "alerts.py",
    "alert_events.py",
]


def run_script(script_name):
    print("\n==============================")
    print(f"Running {script_name}")
    print("==============================")

    subprocess.run(
        [sys.executable, script_name],
        check=True,
    )


def main():
    for script_name in PIPELINE_SCRIPTS:
        try:
            run_script(script_name)
        except subprocess.CalledProcessError as error:
            print("\nPipeline failed.")
            print(f"Failed script: {script_name}")
            print(f"Exit code: {error.returncode}")
            sys.exit(error.returncode)

    print("\n==============================")
    print("Pipeline completed successfully")
    print("==============================")


if __name__ == "__main__":
    main()