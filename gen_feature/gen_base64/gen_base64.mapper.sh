export cur_path="$(cd "`dirname "$0"`"/.; pwd)"
export LD_LIBRARY_PATH=$cur_path:$cur_path/tfdep:$LD_LIBRARY_PATH
cat - | python27/bin/python2.7 gen_base64.mapper.py --conf $1 --gzip_in
