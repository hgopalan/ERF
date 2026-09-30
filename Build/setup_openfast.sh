#!/usr/bin/env bash
# Download, build and install the OpenFAST version ERF's turbine coupling links against.
#
#   Build/setup_openfast.sh [--version 5.0.0] [--prefix DIR] [--src DIR] [--jobs N]
#                           [--fortran gfortran] [--configure-only]
#
# Clones the release tag into <src>/openfast-<version>, configures it the way ERF needs
# (shared libraries, double precision, the C/C++ API on, no tests), builds and installs it under
# <prefix> (default $HOME/opt/openfast-<version>), and prints the -DOPENFAST_DIR to pass to
# ERF's cmake. Re-running is safe: an existing clone is reused and an existing install is left
# alone unless --force is given. The ERF coupling uses the external-inflow C API, which is the
# same in every 4.x and 5.x release; 5.0.0 is the tested version.
set -euo pipefail

version="5.0.0"
src_root="${HOME}/opt/src"
prefix=""
jobs=4
fortran="${FC:-gfortran}"
configure_only=0
force=0
repo="https://github.com/OpenFAST/openfast"

usage() { sed -n '2,13p' "$0"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)        version="$2"; shift 2 ;;
        --prefix)         prefix="$2"; shift 2 ;;
        --src)            src_root="$2"; shift 2 ;;
        --jobs|-j)        jobs="$2"; shift 2 ;;
        --fortran)        fortran="$2"; shift 2 ;;
        --configure-only) configure_only=1; shift ;;
        --force)          force=1; shift ;;
        -h|--help)        usage 0 ;;
        *) echo "setup_openfast.sh: unknown argument '$1'" >&2; usage 1 ;;
    esac
done
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "setup_openfast.sh: --version must look like 5.0.0, not '$version'" >&2; exit 1; }
[[ "${version%%.*}" -ge 4 ]] || { echo "setup_openfast.sh: ERF needs OpenFAST 4 or newer (FAST_ExtInfw_* API), not $version" >&2; exit 1; }
[[ -n "$prefix" ]] || prefix="${HOME}/opt/openfast-${version}"
for tool in git cmake "$fortran"; do
    command -v "$tool" > /dev/null || { echo "setup_openfast.sh: '$tool' not found in PATH" >&2; exit 1; }
done

src="${src_root}/openfast-${version}"
if [[ -d "$src/.git" ]]; then
    echo "setup_openfast.sh: reusing the source in $src"
else
    mkdir -p "$src_root"
    echo "setup_openfast.sh: cloning OpenFAST v${version} into $src"
    git clone --quiet --depth 1 --branch "v${version}" "$repo" "$src"
fi

if [[ $force -eq 0 && -f "$prefix/include/FAST_Library.h" ]] && ls "$prefix"/lib/libopenfastlib.* > /dev/null 2>&1; then
    echo "setup_openfast.sh: OpenFAST ${version} is already installed in $prefix (use --force to rebuild)"
    echo "  ERF: cmake ... -DERF_ENABLE_OPENFAST=ON -DOPENFAST_DIR=$prefix"
    exit 0
fi

build="$src/build"
mkdir -p "$build"
echo "setup_openfast.sh: configuring in $build (install prefix $prefix)"
cmake -S "$src" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_COMPILER="$(command -v "$fortran")" \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DBUILD_SHARED_LIBS=ON \
    -DDOUBLE_PRECISION=ON \
    -DBUILD_OPENFAST_CPP_API=ON \
    -DBUILD_OPENFAST_CPP_DRIVER=OFF \
    -DBUILD_TESTING=OFF \
    -DOPENMP=OFF > "$build/setup_openfast.configure.log" 2>&1 \
    || { tail -30 "$build/setup_openfast.configure.log" >&2; echo "setup_openfast.sh: configure failed (log: $build/setup_openfast.configure.log)" >&2; exit 1; }
if [[ $configure_only -eq 1 ]]; then
    echo "setup_openfast.sh: configured only; build with: cmake --build $build -j $jobs && cmake --install $build"
    exit 0
fi
echo "setup_openfast.sh: building with $jobs jobs (this takes a while; log: $build/setup_openfast.build.log)"
cmake --build "$build" -j "$jobs" > "$build/setup_openfast.build.log" 2>&1 \
    || { tail -30 "$build/setup_openfast.build.log" >&2; echo "setup_openfast.sh: build failed" >&2; exit 1; }
cmake --install "$build" > "$build/setup_openfast.install.log" 2>&1 \
    || { tail -30 "$build/setup_openfast.install.log" >&2; echo "setup_openfast.sh: install failed" >&2; exit 1; }
[[ -f "$prefix/include/FAST_Library.h" ]] || { echo "setup_openfast.sh: the install has no include/FAST_Library.h" >&2; exit 1; }
grep -q "FAST_ExtInfw_Init" "$prefix/include/FAST_Library.h" || { echo "setup_openfast.sh: FAST_Library.h lacks FAST_ExtInfw_Init; ERF needs the OpenFAST 4+ external-inflow API" >&2; exit 1; }
echo "setup_openfast.sh: OpenFAST ${version} installed in $prefix"
echo "  ERF: cmake ... -DERF_ENABLE_OPENFAST=ON -DOPENFAST_DIR=$prefix"
case "$(uname -s)" in
    Darwin) echo "  run with: export DYLD_LIBRARY_PATH=$prefix/lib" ;;
    *)      echo "  run with: export LD_LIBRARY_PATH=$prefix/lib" ;;
esac
