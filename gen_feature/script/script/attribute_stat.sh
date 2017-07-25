INPUT=$1
OUTPUT=$2

cat $INPUT | /data0/python27.luigi/bin/python attribute_stat.py > $OUTPUT
