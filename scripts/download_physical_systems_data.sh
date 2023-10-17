#! /bin/bash

# Taken and adapted from: https://github.com/karlotness/nn-benchmark/blob/master/download.
# EDIT DATA_DIR here:
DATA_DIR="$HOME/data/physical-nn-benchmark"

# Download datasets and experiment outputs from storage in the NYU
# Faculty Digital Archive
#
# This script will download the requested files and if necessary
# recombine multipart files.
#
# Usage: download.sh TYPE SYSTEM DATA_DIR
#
# TYPE is the download type ("run-descriptions" "data" or
# "full")
#
# SYSTEM is the system to download for "data" or "full" TYPEs, options
# are "spring" "wave" "spring-mesh" "navier-stokes-single" or
# "navier-stokes-multi"
#
# If TYPE is "run-descriptions" then SYSTEM should not be specified

set -euo pipefail

download_file() {
    DOWNLOAD_TYPE="$1"
    SYSTEM="$2"
    DATA_DIR="$3"
    case $DOWNLOAD_TYPE in
        "run-descriptions")
            URLS=( 'https://archive.nyu.edu/bitstream/2451/63318/1/nn-benchmark-run-descriptions.tar.gz' )
            OUT_NAME="nn-benchmark-run-descriptions.tar.gz"
            SHASUM="d6460bb3f83f09c8da4b007d46c3c7a0762ad37d217d61f722b3fd82f58efeda"
            ;;
        "data")
            case $SYSTEM in
                "spring")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63316/1/nn-benchmark-data-spring.tar.gz' )
                    OUT_NAME="nn-benchmark-data-spring.tar.gz"
                    SHASUM="1843decc962f6b4616d70be4d174a0138fc872fc651c148f67466043338fdf65"
                    ;;
                "wave")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63316/2/nn-benchmark-data-wave.tar.gz' )
                    OUT_NAME="nn-benchmark-data-wave.tar.gz"
                    SHASUM="2bf5df4f1e883100d420aaed1240d5f32ec2d6d3161aca11209198a8056f53f8"
                    ;;
                "spring-mesh")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63316/3/nn-benchmark-data-spring-mesh.tar.gz' )
                    OUT_NAME="nn-benchmark-data-spring-mesh.tar.gz"
                    SHASUM="548bf16fbbf7dc06fcc29b968a3f8ecb7e81999252c83e41446bc3b975c013a6"
                    ;;
                "navier-stokes-single")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63316/4/nn-benchmark-data-navier-stokes-single.tar.gz' )
                    OUT_NAME="nn-benchmark-data-navier-stokes-single.tar.gz"
                    SHASUM="27fca266d6955d9a0a692842ba728e3100c1c9fe17ef33a12fcf86edec1e6704"
                    ;;
                "navier-stokes-multi")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63316/5/nn-benchmark-data-navier-stokes-multi.tar.gz' )
                    OUT_NAME="nn-benchmark-data-navier-stokes-multi.tar.gz"
                    SHASUM="affa7e210f5eec8d443c4c7cd54afc7632ec65be640a4b6a5dc5d96356f5ce59"
                    ;;
                *)
                    echo "ERROR: Unknown system $SYSTEM"
                    exit 1
                    ;;
            esac
            ;;
        "full")
            case $SYSTEM in
                "spring")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63317/30/nn-benchmark-full-spring.tar.gz' )
                    OUT_NAME="nn-benchmark-full-spring.tar.gz"
                    SHASUM="6c6f4da7fb3cb9af82a805c35c7896eba807b8798f07188a17a65f6261e993b8"
                    ;;
                "wave")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63317/31/nn-benchmark-full-wave.tar.gz.PART00'
                           'https://archive.nyu.edu/bitstream/2451/63317/32/nn-benchmark-full-wave.tar.gz.PART01'
                           'https://archive.nyu.edu/bitstream/2451/63317/33/nn-benchmark-full-wave.tar.gz.PART02'
                           'https://archive.nyu.edu/bitstream/2451/63317/34/nn-benchmark-full-wave.tar.gz.PART03'
                           'https://archive.nyu.edu/bitstream/2451/63317/35/nn-benchmark-full-wave.tar.gz.PART04'
                           'https://archive.nyu.edu/bitstream/2451/63317/36/nn-benchmark-full-wave.tar.gz.PART05'
                           'https://archive.nyu.edu/bitstream/2451/63317/37/nn-benchmark-full-wave.tar.gz.PART06' )
                    OUT_NAME="nn-benchmark-full-wave.tar.gz"
                    SHASUM="1f328b0138f77c3fcba911a75fa0ee32c76febe73343b51804e2d674c3645eb6"
                    ;;
                "spring-mesh")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63317/25/nn-benchmark-full-spring-mesh.tar.gz.PART00'
                           'https://archive.nyu.edu/bitstream/2451/63317/26/nn-benchmark-full-spring-mesh.tar.gz.PART01'
                           'https://archive.nyu.edu/bitstream/2451/63317/27/nn-benchmark-full-spring-mesh.tar.gz.PART02'
                           'https://archive.nyu.edu/bitstream/2451/63317/28/nn-benchmark-full-spring-mesh.tar.gz.PART03'
                           'https://archive.nyu.edu/bitstream/2451/63317/29/nn-benchmark-full-spring-mesh.tar.gz.PART04' )
                    OUT_NAME="nn-benchmark-full-spring-mesh.tar.gz"
                    SHASUM="a0c918e8c96093b5a94af98a230e09ba00229f8a46c7932210e90c1da329c2da"
                    ;;
                "navier-stokes-single")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63317/13/nn-benchmark-full-navier-stokes-single.tar.gz.PART00'
                           'https://archive.nyu.edu/bitstream/2451/63317/14/nn-benchmark-full-navier-stokes-single.tar.gz.PART01'
                           'https://archive.nyu.edu/bitstream/2451/63317/15/nn-benchmark-full-navier-stokes-single.tar.gz.PART02'
                           'https://archive.nyu.edu/bitstream/2451/63317/16/nn-benchmark-full-navier-stokes-single.tar.gz.PART03'
                           'https://archive.nyu.edu/bitstream/2451/63317/17/nn-benchmark-full-navier-stokes-single.tar.gz.PART04'
                           'https://archive.nyu.edu/bitstream/2451/63317/18/nn-benchmark-full-navier-stokes-single.tar.gz.PART05'
                           'https://archive.nyu.edu/bitstream/2451/63317/19/nn-benchmark-full-navier-stokes-single.tar.gz.PART06'
                           'https://archive.nyu.edu/bitstream/2451/63317/20/nn-benchmark-full-navier-stokes-single.tar.gz.PART07'
                           'https://archive.nyu.edu/bitstream/2451/63317/21/nn-benchmark-full-navier-stokes-single.tar.gz.PART08'
                           'https://archive.nyu.edu/bitstream/2451/63317/22/nn-benchmark-full-navier-stokes-single.tar.gz.PART09'
                           'https://archive.nyu.edu/bitstream/2451/63317/23/nn-benchmark-full-navier-stokes-single.tar.gz.PART10'
                           'https://archive.nyu.edu/bitstream/2451/63317/24/nn-benchmark-full-navier-stokes-single.tar.gz.PART11' )
                    OUT_NAME="nn-benchmark-full-navier-stokes-single.tar.gz"
                    SHASUM="d757fc803d55301e32f092168538e0847c89f1be8dc3fa5e89e0176528b801b9"
                    ;;
                "navier-stokes-multi")
                    URLS=( 'https://archive.nyu.edu/bitstream/2451/63317/1/nn-benchmark-full-navier-stokes-multi.tar.gz.PART00'
                           'https://archive.nyu.edu/bitstream/2451/63317/2/nn-benchmark-full-navier-stokes-multi.tar.gz.PART01'
                           'https://archive.nyu.edu/bitstream/2451/63317/3/nn-benchmark-full-navier-stokes-multi.tar.gz.PART02'
                           'https://archive.nyu.edu/bitstream/2451/63317/4/nn-benchmark-full-navier-stokes-multi.tar.gz.PART03'
                           'https://archive.nyu.edu/bitstream/2451/63317/5/nn-benchmark-full-navier-stokes-multi.tar.gz.PART04'
                           'https://archive.nyu.edu/bitstream/2451/63317/6/nn-benchmark-full-navier-stokes-multi.tar.gz.PART05'
                           'https://archive.nyu.edu/bitstream/2451/63317/7/nn-benchmark-full-navier-stokes-multi.tar.gz.PART06'
                           'https://archive.nyu.edu/bitstream/2451/63317/8/nn-benchmark-full-navier-stokes-multi.tar.gz.PART07'
                           'https://archive.nyu.edu/bitstream/2451/63317/9/nn-benchmark-full-navier-stokes-multi.tar.gz.PART08'
                           'https://archive.nyu.edu/bitstream/2451/63317/10/nn-benchmark-full-navier-stokes-multi.tar.gz.PART09'
                           'https://archive.nyu.edu/bitstream/2451/63317/11/nn-benchmark-full-navier-stokes-multi.tar.gz.PART10'
                           'https://archive.nyu.edu/bitstream/2451/63317/12/nn-benchmark-full-navier-stokes-multi.tar.gz.PART11' )
                    OUT_NAME="nn-benchmark-full-navier-stokes-multi.tar.gz"
                    SHASUM="41a4a1f4cc9d052d3851eb983c172f2041056090dbd9e0bf9d22dc6efe8b9fc0"
                    ;;
                *)
                    echo "ERROR: Unknown system $SYSTEM"
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "ERROR: Unknown download type $DOWNLOAD_TYPE"
            exit 1
            ;;
    esac

    # DOWNLOAD TO $DATA_DIR
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    echo "Downloading $DOWNLOAD_TYPE $SYSTEM to $DATA_DIR"
    if [[ -e "$OUT_NAME" ]]; then
        echo "ERROR: File $OUT_NAME already exists, will not overwrite."
        exit 2
    fi

    # Continue with the download
    DL_SHASUM=$(curl --insecure "${URLS[@]}" | tee "$OUT_NAME" | sha256sum | cut -d" " -f1 | tr '[:upper:]' '[:lower:]')

    # Check the hash
    if [[ "$DL_SHASUM" != "$SHASUM" ]]; then
        echo "ERROR: Checksum does not match!"
        rm "$OUT_NAME"
        exit 3
    fi

    # Unpack the file
    echo $OUT_NAME
    tar -xzf "$OUT_NAME"

    # Remove the tar file
    rm "$OUT_NAME"

    echo "Download finished, result stored in $OUT_NAME"
}

show_help() {
    echo "download.sh TYPE SYSTEM"
    echo ""
    echo "TYPE - The type of file to download"
    echo "  options: run-descriptions data full"
    echo ""
    echo "SYSTEM - The system to download (not applicable for run-descriptions)"
    echo "  options: spring wave spring-mesh navier-stokes-single navier-stokes-multi"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    show_help
    exit 0
fi
TYPE="$1"

SYSTEM="n/a"
if [[ $# -lt 2 && "$TYPE" != "run-descriptions" ]]; then
    show_help
    exit 0
elif [[ "$TYPE" != "run-descriptions" ]]; then
    SYSTEM="$2"
fi

download_file "$TYPE" "$SYSTEM" "$DATA_DIR"