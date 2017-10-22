# ./resize_images.sh /path/to/input/folder/ /path/to/output/folder/ quality resize
# Example of usage:
# ./resize_images.sh ./dataset/original/jpg/ ./dataset/down_2x/  100 50
USAGE="USAGE: ./resize_images.sh /path/to/input/folder/ /path/to/output/folder/ quality resize"
EXAMPLE="EXAMPLE: ./resize_images.sh ./dataset/original/jpg/ ./dataset/down_2x/  100 50"

if [ $# -lt 4 ]
	then
		echo "Missing arguments!!!"
		echo
		echo ${USAGE}
		echo ${EXAMPLE}
		exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
QUALITY=$3
RESIZE=$4

echo "Converting images from "${INPUT_DIR} " to " ${OUTPUT_DIR} " with " ${QUALITY}"% quality and " ${RESIZE}"% resize..."

find ${INPUT_DIR} -name '*.jpg' -exec basename {} \; | parallel convert -quality ${QUALITY}% -resize ${RESIZE}% ${INPUT_DIR}/{} ${OUTPUT_DIR}/{}_res_${RESIZE}.jpg
