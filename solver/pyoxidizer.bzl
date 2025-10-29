# This is the full and complete pyoxidizer.bzl file.

def make_exe():
    dist = default_python_distribution(flavor="standalone_dynamic")

    # Create the policy
    policy = dist.make_python_packaging_policy()
    
    # 1. This fixes the "__file__" error by creating a 'lib' directory.
    policy.resources_location = "filesystem-relative:lib"

    # --- THIS IS THE NEW FIX ---
    # 2. Tell the scanner to find all files (.py, .dll, etc.).
    #    (We REMOVED 'file_scanner_classify_files = False').
    policy.file_scanner_emit_files = True
    
    # 3. Allow these generic files to be processed.
    policy.allow_files = True
    
    # 4. This is the new, critical line:
    #    Tell the packager to INCLUDE the generic files it found.
    policy.include_file_resources = True
    # --- END FIX ---


    python_config = dist.make_python_interpreter_config()
    python_config.run_module = "main"

    exe = dist.to_python_executable(
        name = "solver",
        packaging_policy = policy, # Pass the fully configured policy
        config = python_config,
    )

    # Install dependencies from requirements.txt
    exe.add_python_resources(
        exe.pip_install(["-r", "../requirements.txt"])
    )

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