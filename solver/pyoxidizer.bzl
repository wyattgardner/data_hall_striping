# This is the full and complete pyoxidizer.bzl file.

def make_exe():
    dist = default_python_distribution()

    # Create the policy
    policy = dist.make_python_packaging_policy()
    
    # 1. Set to "in-memory" for a single-file .exe
    policy.resources_location = "in-memory"

    # --- FIX #1: Solve the runtime __file__ error ---
    python_config = dist.make_python_interpreter_config()
    python_config.run_module = "main"

    # This tells the interpreter to unpack C-extensions to disk
    # before importing, which fixes the NumPy/OR-Tools crash.
    python_config.prefer_in_memory_imports = False
    # --- END FIX #1 ---

    exe = dist.to_python_executable(
        name = "solver",
        packaging_policy = policy, # Pass the configured policy
        config = python_config,   # Pass the configured config
    )

    # --- FIX #2: Embed the necessary .dll/.pyd files ---
    # Install dependencies from requirements.txt
    # 'include_non_python_files=True' is the correct syntax for v0.24.0
    # to find and embed the files NumPy and OR-Tools need.
    exe.add_python_resources(
        exe.pip_install(
            ["-r", "../requirements.txt"],
            include_non_python_files=True
        )
    )
    # --- END FIX #2 ---

    # Include your main app code
    exe.add_python_resources(
        exe.read_package_root(
            path = "../",
            packages = ["main"], # This finds your main.py
        )
    )

    return exe


def make_install(exe):
    files = FileManifest()
    files.add_python_resource(".", exe)
    return files


register_target("exe", make_exe)
register_target("install", make_install, depends=["exe"], default=True)

resolve_targets()