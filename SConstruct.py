# SConstruct - SCons build script for Godot GDExtension (Native Windows API)
#
# How to run (from your terminal in this directory):
# For a debug build: scons
# For a release build: scons target=template_release

import os

# --- SCons Setup ---
# This line is essential. It runs the SConstruct file from godot-cpp,
# which sets up the core environment for building the GDExtension.
env = SConscript("godot-cpp/SConstruct")

# --- Configuration ---
# You can change your library name here
library_name = "shared_memory_link"
# --------------------

# --- Add Source Path ---
# Tell SCons where to find our own source code's header files.
env.Append(CPPPATH=["src"])

# --- Target and Path Setup ---
## FIX: Use the correct Godot target names ('template_debug', 'template_release')
## and set a default that matches.
target = ARGUMENTS.get("target", "template_debug")

## FIX: Initialize lib_path outside the platform-specific block to avoid scope errors.
lib_path = ""
if "debug" in target:
    lib_path = "game/shared-memory-link/bin/debug"
elif "release" in target:
    lib_path = "game/shared-memory-link/bin/release"
else:
    print(f"Error: Unknown target '{target}'. Use 'template_debug' or 'template_release'.")
    Exit(1)


## CRITICAL FIX: The line `env = Environment(...)` has been removed.
## We must NOT overwrite the environment created by godot-cpp's SConstruct.
## All modifications should be made to the existing 'env' object.

# --- Platform and Compiler Configuration ---
if env["platform"] == "windows":
    print(f"Configuring for Windows platform, target: {target}")

    # --- Link against Windows-specific libraries ---
    # This is required for all Windows builds using the Windows API.
    env.Append(LIBS=['user32'])

    ## FIX FOR WARNING D9025:
    ## Remove any existing C++ standard flag set by godot-cpp.
    ## This is a list comprehension that rebuilds the list without the unwanted flag.
    env['CXXFLAGS'] = [flag for flag in env['CXXFLAGS'] if not flag.startswith('/std:c++')]



    # The godot-cpp script already sets up most necessary flags. We only need to add our own.
    if "debug" in target:
        print("Appending MSVC Debug compiler options.")
        # Append specific MSVC compiler flags for debugging.
        # /EHsc is often included by default in MSVC, but it's safe to add.
        # godot-cpp already adds /Zi, /DDEBUG_ENABLED, and /MDd (debug runtime).
        env.Append(
            CXXFLAGS=[
                "/EHsc",
                "/FS",          # Force synchronous PDB writes.
                "/std:c++20",   # Use the C++20 standard.
            ]
        )
        # Linker flags for debugging.
        env.Append(LINKFLAGS=["/DEBUG"])

    elif "release" in target:
        print("Appending MSVC Release compiler options.")
        # godot-cpp already adds /O2 and /MD (release runtime).
        # We just need to add our specific requirements.
        env.Append(
            CXXFLAGS=[
                "/std:c++20",   ## FIX: Corrected typo from /std:c++1
            ]
        )
        ## NOTE: Flags like /O2 and /MD are already handled by the godot-cpp SConstruct
        ## for release builds. Adding them again is redundant.
        ## The NDEBUG define is also typically handled.

# --- Compile Our Source Files ---
# Find all .cpp files in the 'src' directory.
sources = Glob("src/*.cpp")

# --- Build the Final Library ---
# This command tells SCons to build the final shared library.
# It links our source code with the compiled godot-cpp library.
## FIX: The output path construction now uses the correctly-scoped `lib_path`.
library = env.SharedLibrary(
    target=os.path.join(lib_path, f"{library_name}{env['suffix']}{env['SHLIBSUFFIX']}"),
    source=sources, # SCons will compile these sources into objects automatically.
)

# Set this library as the default target to build when you just run `scons`
Default(library)

# Add a "clean" target so "scons -c" works correctly.
# This will remove the final library and its associated files.
Clean(library, os.path.join(lib_path, f"{library_name}.pdb"))
Clean(library, os.path.join(lib_path, f"{library_name}.exp"))
Clean(library, os.path.join(lib_path, f"{library_name}.lib"))