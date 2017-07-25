set -x 
name=${0%*.sh}
input=$1
output=$2
reducer_number=0
if [[ $# -gt 2 ]]; then
	reducer_number=$3
fi

mapper="${name}.mapper.sh"
mapper_script_path="${name}.mapper.py"
conf="${name}.mapred.conf"

hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar    \
	-D mapred.job.priority=VERY_HIGH     \
	-D mapred.job.name="$name"      \
    -D mapred.map.tasks=5000 \
    -D mapred.reduce.tasks=$reducer_number \
    -D mapreduce.map.memory.mb=1512 \
    -D mapreduce.reduce.memory.mb=2048 \
	-cacheArchive "/user/jd_ad/bailu/archieve/tensorflow2.tar.gz#python27" \
	-cacheArchive "/user/jd_ad/bailu/archieve/tfdep2.so.tar.gz#tfdep"		\
	-input "${input}"       \
	-output "${output}"     \
	-mapper "sh $mapper $conf" -file "$mapper" -file "$mapper_script_path" -file "tfcrc.so" -file "$conf" \
	-reducer NONE

