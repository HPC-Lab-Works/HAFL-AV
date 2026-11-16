#!/usr/bin/env bash
# Downloads the KITTI 2D object-detection benchmark (images + labels).
#
# KITTI requires a free account at https://www.cvlibs.net/datasets/kitti/
# before these links will serve files (an anonymous `wget`/`curl` gets a
# login-redirect HTML page instead of the zip, not a clear error, so this
# script checks the downloaded file size and tells you explicitly if that
# happened instead of silently producing a corrupt dataset).
#
# Usage:
#   bash data/download_kitti.sh /path/to/data_raw/kitti
#
# After registering, log in once in a browser, then either:
#   (a) pass --cookies <netscape-cookie-file> exported from that browser session, or
#   (b) download the two zips manually from the "2D object" section of the
#       KITTI website and place them at <out_dir>/raw/ before re-running this
#       script with --skip-download to just unpack + checksum what's there.

set -euo pipefail

OUT_DIR="${1:?Usage: download_kitti.sh <out_dir> [--cookies FILE] [--skip-download]}"
shift || true
COOKIES=""
SKIP_DOWNLOAD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cookies) COOKIES="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"

IMG_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
LBL_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

CURL_OPTS=(--fail --location --continue-at - -o)
if [[ -n "${COOKIES}" ]]; then
  CURL_OPTS=(--fail --location --cookie "${COOKIES}" --continue-at - -o)
fi

download_one () {
  local url="$1" dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "[skip] ${dest} already exists"
    return
  fi
  echo "[download] ${url} -> ${dest}"
  curl "${CURL_OPTS[@]}" "${dest}" "${url}"
  # A login-wall redirect lands here as a small HTML page, not a zip -- a
  # real KITTI zip is hundreds of MB, so anything under 1 MB is almost
  # certainly a failed/auth-gated download, not a real dataset file.
  local size
  size=$(stat -c%s "${dest}" 2>/dev/null || stat -f%z "${dest}")
  if [[ "${size}" -lt 1000000 ]]; then
    echo "ERROR: ${dest} is only ${size} bytes -- this is almost certainly a" >&2
    echo "login page, not the dataset. Register at https://www.cvlibs.net/datasets/kitti/," >&2
    echo "export your browser's cookies, and re-run with --cookies <file>." >&2
    rm -f "${dest}"
    exit 1
  fi
}

if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  download_one "${IMG_URL}" "${RAW_DIR}/data_object_image_2.zip"
  download_one "${LBL_URL}" "${RAW_DIR}/data_object_label_2.zip"
fi

echo "[unzip] images ..."
unzip -n -q "${RAW_DIR}/data_object_image_2.zip" -d "${OUT_DIR}"
echo "[unzip] labels ..."
unzip -n -q "${RAW_DIR}/data_object_label_2.zip" -d "${OUT_DIR}"

echo "Done. Raw KITTI layout now under ${OUT_DIR}/training/{image_2,label_2}."
echo "Next: python data/prepare_kitti.py --raw_dir ${OUT_DIR} --out_dir <yolo_dataset_dir>"
